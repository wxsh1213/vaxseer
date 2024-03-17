from collections import defaultdict
import torch
import numpy as np

def discretize_time(time, one_step, normalize_time_a, normalize_time_b=0, discrete=True):
    if one_step:
        return torch.argsort(time)
    else:
        normalized_time = (time - normalize_time_b) / (normalize_time_a)
        if discrete:
            return torch.ceil(normalized_time)
        else:
            return normalized_time

def year_month_to_int(input, start_date="2019-12"):
    year, month = input.split("-")
    ref_year, ref_month = start_date.split("-")
    return (int(year) - int(ref_year)) * 12 + int(month) - int(ref_month)

def collate_properties(list_of_dict, keys, properties_dict, return_dict=None):
    if not return_dict:
        return_dict = {}
    for key in keys:
        if isinstance(list_of_dict[0][key], np.int64) or \
            isinstance(list_of_dict[0][key], np.float64) or\
                isinstance(list_of_dict[0][key], float) or isinstance(list_of_dict[0][key], int):
            return_dict[key] = torch.tensor([x[key] for x in list_of_dict])
        elif key in properties_dict:
            return_dict[key] = torch.tensor([properties_dict[key][x[key]] for x in list_of_dict])
        elif isinstance(list_of_dict[0][key], list):
            return_dict[key] = torch.tensor([list_of_dict[i][key] for i in range(len(list_of_dict))])
    return return_dict

def default_lm_collate_func(list_of_dict, batch_converter, padding_idx=None, properties_dict={}):
    # src_seqs = [("", remove_redundant_gaps(x["src_seq"])) for x in list_of_dict]
    src_seqs = []
    for x in list_of_dict:
        if isinstance(x["src_seq"], str):
            src_seqs.append(("", remove_redundant_gaps(x["src_seq"])))
        elif isinstance(x["src_seq"], list):
            src_seqs.append([("", y) for y in x["src_seq"]])


    # src_seqs = [("", remove_redundant_gaps(x["src_seq"])) for x in list_of_dict]
    _, _, src_tokens = batch_converter(src_seqs)

    if padding_idx is not None:
        attention_mask = (src_tokens != padding_idx).float()
    else:
        attention_mask = src_tokens.new_ones(src_tokens.size())
    
    ret = {
        "input_ids": src_tokens, # [B x L]
        # "input_time": src_time, # [B]
        "labels": src_tokens, # [B x L]
        "attention_mask": attention_mask # [B x L]
    }
    if "src_time" in list_of_dict[0]:
        src_time = torch.tensor([x["src_time"] if x["src_time"] is not None else 1.0 for x in list_of_dict])
        ret["input_time"] = src_time
        
    other_keys = [x for x in list_of_dict[0].keys() if x != "src_time" and x != "src_seq"]
    
    ret = collate_properties(list_of_dict, other_keys, properties_dict, return_dict=ret)
    
    # for key in other_keys:
    #     if isinstance(list_of_dict[0][key], np.int64) or \
    #         isinstance(list_of_dict[0][key], np.float64) or\
    #             isinstance(list_of_dict[0][key], float) or isinstance(list_of_dict[0][key], int):
    #         ret[key] = torch.tensor([x[key] for x in list_of_dict])
    #     elif key in properties_dict:
    #         ret[key] = torch.tensor([properties_dict[key][x[key]] for x in list_of_dict])
    # # print(ret)
    # # print(properties_dict)

    return ret

def default_msa_collate_func(list_of_dict, batch_converter, padding_idx=None, max_msa_seq_num=128):    
    # tgt_seqs = [x["tgt_seq"] for x in list_of_dict]
    # NOTE: remove extra gaps in target tokens.
    tgt_seqs_flattern = [[(y[0], remove_redundant_gaps(y[1]))] for x in list_of_dict for y in x["tgt_seq"]]
    _, _, tgt_tokens = batch_converter(tgt_seqs_flattern)
    tgt_tokens = tgt_tokens.view(len(list_of_dict), -1, tgt_tokens.size(-1)) # [batch_size x msa_seq_num x max_seq_length]
    # print(np.reshape(np.asarray(a), (3, 15)))
    tgt_time = torch.tensor([x['tgt_time'] for x in list_of_dict]) # [batch_size]
    tgt_msa_masks = torch.tensor(
        [[1] * len(x["tgt_seq"]) + [0] * (tgt_tokens.size(1) - len(x["tgt_seq"]))  for x in list_of_dict],
        dtype=torch.bool
        )
    # print(tgt_time)
    # print(tgt_tokens.size(), tgt_time.size())

    # build sources
    src_seqs = [] # [x["src_seq"][0] for x in list_of_dict] # len(src_seqs[i]) <= window size, src_seqs[i][j]: blocks in each time
    src_times = []
    src_msa_masks = [] # [B x T x max_msa_num]
    for x in list_of_dict:
        _, _, tokens = batch_converter(x["src_seq"][0]) # [T x M x L]
        src_seqs.append(tokens)
        src_times.append(x["src_time"][0])
        src_msa_masks.append(torch.tensor([[1] * len(y) + [0] * (tokens.size(1) - len(y)) for y in x["src_seq"][0]])) # [T x M]
        if len(x["src_seq"]) == 2:
            _, _, tokens = batch_converter(x["src_seq"][1]) # [T x M x L]
            src_seqs.append(tokens)
            src_times.append(x["src_time"][1])
            src_msa_masks.append(torch.tensor([[1] * len(y) + [0] * (tokens.size(1) - len(y)) for y in x["src_seq"][1]]))

    src_seqs_tensor = torch.zeros(len(src_seqs), max([x.size(0) for x in src_seqs]), max([x.size(1) for x in src_seqs]), max([x.size(-1) for x in src_seqs]), dtype=torch.long)
    src_seqs_tensor.fill_(padding_idx)
    src_times_tensor = torch.zeros(len(src_times), max([len(x) for x in src_times]))
    src_times_tensor.fill_(-100)
    src_msa_masks_tensor = torch.zeros(len(src_msa_masks), max([x.size(0) for x in src_msa_masks]), max([x.size(1) for x in src_msa_masks]))
    # print("src_times_tensor.size()", src_times_tensor.size())

    for i, tokens in enumerate(src_seqs):
        src_seqs_tensor[i, :tokens.size(0), :tokens.size(1), :tokens.size(2)] = tokens
    for i, time in enumerate(src_times):
        src_times_tensor[i, :len(time)] = torch.tensor(time)
    for i, masks in enumerate(src_msa_masks):
        src_msa_masks_tensor[i, :masks.size(0), :masks.size(1)] = masks

    direction_num = len(list_of_dict[0]["src_seq"])
    # print(direction_num)
    src_seqs_tensor = src_seqs_tensor.view(-1, direction_num, src_seqs_tensor.size(-3), src_seqs_tensor.size(-2), src_seqs_tensor.size(-1)) # [batch, direction_num, window_size, msa_num, seq_len]
    # print("src_times_tensor.size()", src_times_tensor.size())
    src_times_tensor = src_times_tensor.view(-1, direction_num, src_times_tensor.size(-1)) # [batch, direction_num, window_size]
    src_msa_masks_tensor = src_msa_masks_tensor.view(-1, direction_num, src_msa_masks_tensor.size(-2), src_msa_masks_tensor.size(-1))
    # print("src_times_tensor.size()", src_times_tensor.size())
    # print(src_times_tensor)

    ret = {
        "src_tokens": src_seqs_tensor, 
        "src_time": src_times_tensor, 
        "src_msa_masks": src_msa_masks_tensor,
        "tgt_tokens": tgt_tokens,
        "tgt_time": tgt_time,
        "tgt_msa_masks": tgt_msa_masks
    }
    return ret

def remove_redundant_gaps(src_seq, tgt_seq=None, remove_gaps_from_source=False):
    if tgt_seq is not None: 
        if remove_gaps_from_source:
            new_src, new_tgt = [], []
            for src_c, tgt_c in zip(src_seq, tgt_seq):
                if src_c != "-":
                    new_src.append(src_c)
                    new_tgt.append(tgt_c)
            return "".join(new_src), "".join(new_tgt)
        else: # Remove double gaps
            new_src_seq = np.asarray(list(src_seq))
            new_tgt_seq = np.asarray(list(tgt_seq))
            gaps = np.logical_and((new_src_seq == "-"), (new_tgt_seq == "-"))
            return "".join(new_src_seq[~gaps]), "".join(new_tgt_seq[~gaps])
    else:
        if len(src_seq) == 0:
            return src_seq
        new_src_seq = np.asarray(list(src_seq))
        return "".join(new_src_seq[new_src_seq != "-"])

def remove_gaps_from_source(src_tokens, tgt_tokens, gap_idx, pad_idx): # self.alphabet.gap_idx
    # print(src_tokens.size())
    assert src_tokens.size() == tgt_tokens.size()
    gaps_masks = (src_tokens == gap_idx)
    # print(gaps_masks)
    # print(gaps_masks.sum(-1))
    indices = torch.arange(src_tokens.size(1), device=src_tokens.device).unsqueeze(0).repeat(src_tokens.size(0), 1)
    # print(indices.size(), src_tokens.size())
    indices[gaps_masks] = gaps_masks.size(1) # .masked_fill_(gaps_masks, gaps_masks.size(1))
    indices_sorted, indices_sorted_indices = torch.sort(indices, dim=-1)
    # print("indices_sorted_indices", indices_sorted_indices)
    src_tokens_gaps_removed = torch.gather(src_tokens, 1, indices_sorted_indices)
    tgt_tokens_gaps_removed = torch.gather(tgt_tokens, 1, indices_sorted_indices)
    # print(src_tokens_gaps_removed)
    gaps_masks = (src_tokens_gaps_removed == gap_idx)
    # print(gaps_masks)
    # print(gaps_masks.sum(-1))
    src_tokens_gaps_removed[gaps_masks] = pad_idx # self.alphabet.pad()
    tgt_tokens_gaps_removed[gaps_masks] = pad_idx
    # print(src_tokens_gaps_removed)
    # print(tgt_tokens_gaps_removed)
    max_length = (src_tokens_gaps_removed != pad_idx).sum(-1).max().item()
    return src_tokens_gaps_removed[:, :max_length], tgt_tokens_gaps_removed[:, :max_length]

    def __init__(self, xyz1, xyz2):
        # The first one the reference structure
        self.calc_rmsd(xyz1, xyz2)

    def apply(self, xyz):
        xyz = xyz - xyz.mean(axis=0, keepdims=True)
        return np.dot(xyz, self.U)
        
    def calc_rmsd(self, xyz1, xyz2):
        c_trans, U, ref_trans = fit_rms(xyz1, xyz2)
        new_c2 = np.dot(xyz2 - c_trans, U) # + ref_trans
        new_c1 = xyz1 - ref_trans
        rmsd = np.sqrt( np.average( np.sum( ( new_c1 - new_c2 )**2, axis=1 ) ) )
        self.U = U
        self.ref_trans = ref_trans
        return rmsd, new_c1, new_c2

    # def get_aligned_coord(self, xyz, name=None):
    #     new_c2 = deepcopy(xyz)

    #     for atom in new_c2:
    #         atom.x, atom.y, atom.z = np.dot(np.array([atom.x, atom.y, atom.z]) - self.c_trans, self.U) + self.ref_trans
    #     return new_c2