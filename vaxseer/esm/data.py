# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
from typing import Sequence, Tuple, List, Union
import pickle
import re
import shutil
import torch
from pathlib import Path
from .constants import proteinseq_toks, nucleotide_toks, get_codons, valid_codons
import torch.nn.functional as F
import numpy as np

RawMSA = Sequence[Tuple[str, str]]


class FastaBatchedDataset(object):
    def __init__(self, sequence_labels, sequence_strs):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)

    @classmethod
    def from_file(cls, fasta_file):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []

        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []

        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())

        _flush_current_seq()

        assert len(set(sequence_labels)) == len(
            sequence_labels
        ), "Found duplicate sequence labels"

        return cls(sequence_labels, sequence_strs)

    def __len__(self):
        return len(self.sequence_labels)

    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0

        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)

        _flush_current_buf()
        return batches


class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
        gap_tok: str = '-',
        codon=False
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa
        self.codon = codon

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        if gap_tok not in self.all_toks:
            self.all_toks.append(gap_tok)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>', '<sep>']
        self.gap_idx = self.get_idx("-")
        self.unique_no_split_tokens = self.all_toks
        self.gap_tok = gap_tok
    
        self.std_toks_idx = [self.tok_to_idx[i] for i in self.standard_toks]
        self.non_std_toks_idx = [self.tok_to_idx[i] for i in self.all_toks if i not in self.standard_toks]

    def pad(self, ):
        return self.padding_idx
    
    def eos(self, ):
        return self.eos_idx
    
    def bos(self, ):
        return self.cls_idx
    
    def unk(self,):
        return self.unk_idx

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self, use_msa=False, **kwargs):
        # print("get_batch_converter", kwargs)
        if use_msa: # Overwrite, if True...
            return MSABatchConverter(self, **kwargs)
        if self.use_msa:
            return MSABatchConverter(self, **kwargs)
        else:
            return BatchConverter(self, **kwargs)
    
    def concat(self, seq1, seq2):
        if "<sep>" in self.all_toks:
            return seq1 + "<sep>" + seq2
        else:
            return seq1 + "<eos>" + seq2

    @classmethod
    def from_architecture(cls, name: str, mol_type="protein") -> "Alphabet":
        if mol_type == "protein":
            standard_toks = proteinseq_toks["toks"]
        elif mol_type == "nucleotide":
            standard_toks = nucleotide_toks["toks"]
        elif mol_type == "rna_codon":
            standard_toks = get_codons("UCAG")["toks"] # virus
        elif mol_type == "dna_codon":
            standard_toks = valid_codons["toks"]
            # standard_toks = get_codons("TCAG")["toks"] # virus

        if name in ("ESM-1", "protein_bert_base"):
            # standard_toks = proteinseq_toks["toks"]
            prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        elif name in ("ESM-1b", "roberta_large"):
            # standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
        elif name == "our":
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>", "*")
            append_toks = ("<mask>", )
            prepend_bos = True
            append_eos = True
            use_msa = False
        elif name in ("MSA Transformer", "msa_transformer"):
            # standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = False
            use_msa = True
        elif name in ("t5-base", "gpt2"):
            # standard_toks = proteinseq_toks["toks"]
            # prepend_toks = ("<pad>", "<eos>", "<cls>", "<unk>")
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>", "<sep>")
            # prepend_bos = False
            prepend_bos = True
            append_eos = True
            use_msa = False
        elif "invariant_gvp" in name.lower():
            # standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>", "<cath>", "<af2>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        else:
            raise ValueError("Unknown architecture selected: %s" % name)
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa, codon=mol_type in ("dna_codon", "rna_codon"))

    def _tokenize(self, text) -> str:
        return text.split()

    def __call__(self, text, **kwds):
        return self.tokenize(text, **kwds)

    def tokenize_codon(self, text, **kwargs) -> List[str]: 
        assert len(text) % 3 == 0
        list_of_tokens = []
        for i in range(len(text))[::3]:
            list_of_tokens.append(text[i:i+3])
        return list_of_tokens

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        # if tokenized_text[-1] == "*": # Ignore the "*"
        #     tokenized_text = tokenized_text[:-1]
        return tokenized_text

    # def encode_codon(self, text):
    #     new_tokens = []
    #     tokens = self.tokenize(text)
    #     i = 0
    #     while (i < len(tokens)):
    #         if tokens[i:i+3] in self.standard_toks:
    #             new_tokens.append(self.tok_to_idx["".join(tokens[i:i+3])])
    #         elif tokens[i] in self.


    def encode(self, text):
        if getattr(self, "codon", False):
        # self.codon:
            return [(self.tok_to_idx[tok] if tok in self.tok_to_idx else self.unk_idx) for tok in self.tokenize_codon(text)]
        else:
            return [(self.tok_to_idx[tok] if tok in self.tok_to_idx else self.unk_idx) for tok in self.tokenize(text)]
    
    def encode_line(self, text, **kwargs):
        return torch.tensor(self.encode(text))

    def string(self, indices, remove_special_tokens=True, remove_gaps=True):
        indices = [x for x in indices if x != self.pad()] # remove padding
        # assert indices[0] == self.bos()
        # assert indices[-1] == self.eos(), "Value %d" % indices[-1]
        if indices[0] == self.bos(): # remove bos
            indices = indices[1:]
        if indices[-1] == self.eos(): # remove eos
            indices = indices[:-1]
        if remove_gaps: # remove gaps
            indices = [x for x in indices if x != self.gap_idx]
        tokens = [self.get_tok(idx) for idx in indices]
        if remove_special_tokens:
            tokens = [x for x in tokens if x not in self.all_special_tokens]
        return "".join(tokens)
    
    def remove_gap(self, string):
        return "".join([c for c in string if c != self.gap_tok])


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet, max_positions=None):
        self.alphabet = alphabet
        self.max_positions = max_positions

    def __call__(self, raw_batch: Sequence[Tuple[str, str]], seq_encoded_list=None):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        if seq_encoded_list is None:
            batch_labels, seq_str_list = zip(*raw_batch)
            seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        else:
            seq_str_list = [""] * len(seq_encoded_list)
            batch_labels = [""] * len(seq_encoded_list)

        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        if self.max_positions is not None:
            max_len = min(max_len, self.max_positions - int(self.alphabet.prepend_bos) - int(self.alphabet.append_eos))
        tokens = torch.empty(
            (
                batch_size,
                max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
            zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq_encoded = seq_encoded[:max_len]
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens


class MSABatchConverter(BatchConverter):
    def __call__(self, inputs: Union[Sequence[RawMSA], RawMSA]):
        if isinstance(inputs[0][0], str):
            # Input is a single MSA
            raw_batch: Sequence[RawMSA] = [inputs]  # type: ignore
        else:
            raw_batch = inputs  # type: ignore

        batch_size = len(raw_batch)
        max_alignments = max(len(msa) for msa in raw_batch)
        max_seqlen = max(len(msa[0][1]) for msa in raw_batch)
        if self.max_positions is not None:
            max_seqlen = min(max_seqlen, self.max_positions - int(self.alphabet.prepend_bos) - int(self.alphabet.append_eos))

        tokens = torch.empty(
            (
                batch_size,
                max_alignments,
                max_seqlen + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos),
            ),
            dtype=torch.int64,
        )
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        strs = []

        for i, msa in enumerate(raw_batch):
            msa_seqlens = set(len(seq) for _, seq in msa)
            if not len(msa_seqlens) == 1:
                raise RuntimeError(
                    "Received unaligned sequences for input to MSA, all sequence "
                    "lengths must be equal."
                )
            if isinstance(msa[0], list):
                seq_encoded_list = msa
            else:
                seq_encoded_list = None
            msa_labels, msa_strs, msa_tokens = super().__call__(msa, seq_encoded_list=seq_encoded_list)
            labels.append(msa_labels)
            strs.append(msa_strs)
            tokens[i, : msa_tokens.size(0), : msa_tokens.size(1)] = msa_tokens

        return labels, strs, tokens

class CoordBatchConverter(BatchConverter):
    def __init__(self, alphabet, max_positions=None):
        super().__init__(alphabet, max_positions)
        self.alphabet.append_eos = True
        self.alphabet.prepend_bos = True

    def __call__(self, raw_batch, device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x 3 x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>") 
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))

        coords_and_confidence, strs, tokens = super().__call__(batch)

        # pad beginning and end of each protein due to legacy reasons
        coords = [
            F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.inf)
            for cd, _ in coords_and_confidence
        ]
        # print(len(coords), [len(coords[i]) for i in range(len(coords))], tokens.size())
        # print(coords_and_confidence[0][0].shape, len(strs[0]))
        # print(self.alphabet.prepend_bos, self.alphabet.append_eos)
        # exit()
        confidence = [
            F.pad(torch.tensor(cf), (1, 1), value=-1.)
            for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
        padding_mask = torch.isnan(coords[:,:,0,0])
        coord_mask = torch.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.) * padding_mask
        return coords, confidence, strs, tokens, padding_mask

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result



def read_fasta(
    path,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    with open(path, "r") as f:
        for result in read_alignment_lines(
            f, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
        ):
            yield result


def read_alignment_lines(
    lines,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    seq = desc = None

    def parse(s):
        if not keep_gaps:
            s = re.sub("-", "", s)
        if not keep_insertions:
            s = re.sub("[a-z]", "", s)
        return s.upper() if to_upper else s

    for line in lines:
        # Line may be empty if seq % file_line_width == 0
        if len(line) > 0 and line[0] == ">":
            if seq is not None:
                yield desc, parse(seq)
            desc = line.strip()
            seq = ""
        else:
            assert isinstance(seq, str)
            seq += line.strip()
    assert isinstance(seq, str) and isinstance(desc, str)
    yield desc, parse(seq)


class ESMStructuralSplitDataset(torch.utils.data.Dataset):
    """
    Structural Split Dataset as described in section A.10 of the supplement of our paper.
    https://doi.org/10.1101/622803

    We use the full version of SCOPe 2.07, clustered at 90% sequence identity,
    generated on January 23, 2020.

    For each SCOPe domain:
        - We extract the sequence from the corresponding PDB file
        - We extract the 3D coordinates of the Carbon beta atoms, aligning them
          to the sequence. We put NaN where Cb atoms are missing.
        - From the 3D coordinates, we calculate a pairwise distance map, based
          on L2 distance
        - We use DSSP to generate secondary structure labels for the corresponding
          PDB file. This is also aligned to the sequence. We put - where SSP
          labels are missing.

    For each SCOPe classification level of family/superfamily/fold (in order of difficulty),
    we have split the data into 5 partitions for cross validation. These are provided
    in a downloaded splits folder, in the format:
            splits/{split_level}/{cv_partition}/{train|valid}.txt
    where train is the partition and valid is the concatentation of the remaining 4.

    For each SCOPe domain, we provide a pkl dump that contains:
        - seq    : The domain sequence, stored as an L-length string
        - ssp    : The secondary structure labels, stored as an L-length string
        - dist   : The distance map, stored as an LxL numpy array
        - coords : The 3D coordinates, stored as an Lx3 numpy array

    """

    base_folder = "structural-data"
    file_list = [
        #  url  tar filename   filename      MD5 Hash
        (
            "https://dl.fbaipublicfiles.com/fair-esm/structural-data/splits.tar.gz",
            "splits.tar.gz",
            "splits",
            "456fe1c7f22c9d3d8dfe9735da52411d",
        ),
        (
            "https://dl.fbaipublicfiles.com/fair-esm/structural-data/pkl.tar.gz",
            "pkl.tar.gz",
            "pkl",
            "644ea91e56066c750cd50101d390f5db",
        ),
    ]

    def __init__(
        self,
        split_level,
        cv_partition,
        split,
        root_path=os.path.expanduser("~/.cache/torch/data/esm"),
        download=False,
    ):
        super().__init__()
        assert split in [
            "train",
            "valid",
        ], "train_valid must be 'train' or 'valid'"
        self.root_path = root_path
        self.base_path = os.path.join(self.root_path, self.base_folder)

        # check if root path has what you need or else download it
        if download:
            self.download()

        self.split_file = os.path.join(
            self.base_path, "splits", split_level, cv_partition, f"{split}.txt"
        )
        self.pkl_dir = os.path.join(self.base_path, "pkl")
        self.names = []
        with open(self.split_file) as f:
            self.names = f.read().splitlines()

    def __len__(self):
        return len(self.names)

    def _check_exists(self) -> bool:
        for (_, _, filename, _) in self.file_list:
            fpath = os.path.join(self.base_path, filename)
            if not os.path.exists(fpath) or not os.path.isdir(fpath):
                return False
        return True

    def download(self):

        if self._check_exists():
            print("Files already downloaded and verified")
            return

        from torchvision.datasets.utils import download_url

        for url, tar_filename, filename, md5_hash in self.file_list:
            download_path = os.path.join(self.base_path, tar_filename)
            download_url(url=url, root=self.base_path, filename=tar_filename, md5=md5_hash)
            shutil.unpack_archive(download_path, self.base_path)

    def __getitem__(self, idx):
        """
        Returns a dict with the following entires
         - seq : Str (domain sequence)
         - ssp : Str (SSP labels)
         - dist : np.array (distance map)
         - coords : np.array (3D coordinates)
        """
        name = self.names[idx]
        pkl_fname = os.path.join(self.pkl_dir, name[1:3], f"{name}.pkl")
        with open(pkl_fname, "rb") as f:
            obj = pickle.load(f)
        return obj
