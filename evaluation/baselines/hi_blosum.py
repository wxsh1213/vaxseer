from Bio.Align import substitution_matrices
from Bio import Align, SeqIO
import pandas as pd
import argparse
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np

aligner = Align.PairwiseAligner()
substitution_matrices.load()

matrix = substitution_matrices.load("BLOSUM62")

aligner.substitution_matrix = matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluating the generation performance')
    parser.add_argument('--hi_form_path', default="/data/rsg/nlp/wenxian/esm/data/who_flu/before_2012-02/a_h3n2_hi_folds.csv", type=str)
    parser.add_argument('--sequence_file', default="/data/rsg/nlp/wenxian/esm/data/gisaid/flu/ha.fasta", type=str)
    parser.add_argument('--index_pair', default="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/pipeline/2012-02/a_h3n2/vaccine_set=2009-02-2012-02___virus_set=2009-02-2012-02/vaccine_virus_pairs/pairs.csv", type=str)
    parser.add_argument('--save_path', default="tmp.csv", type=str)
    parser.add_argument('--no_imputation', action="store_true")
    args = parser.parse_args()
    return args

def read_fasta(path):
    accid2seq = {}
    seq2strain_name = defaultdict(set)
    for record in SeqIO.parse(path, "fasta"):
        descs = record.description.split("|")
        try:
            # time = int(descs[5].split("-")[0])
            
            accid = None
            for x in descs:
                if "EPI" in x and "EPI_ISL" not in x:
                    accid = x
            
            assert accid is not None

            accid2seq[accid] = str(record.seq)
            seq2strain_name[str(record.seq)].add(descs[1])
        except Exception as e:
            continue
    return accid2seq, seq2strain_name # , id2record, id2accid

if __name__ == "__main__":
    args = parse_args()

    accid2seq, seq2strain_names = read_fasta(args.sequence_file)

    # read all HI pairs
    seq_pairs2hi_scores = defaultdict(list)
    df_hi = pd.read_csv(args.hi_form_path)
    print(len(df_hi))

    virus_seqs = set()
    vaccine_seqs = set()
    for virus_id, reference_id, hi_value in zip(df_hi["virus"], df_hi["reference"], df_hi["hi"]):
        virus_seq = accid2seq[virus_id]
        reference_seq = accid2seq[reference_id]
        seq_pairs2hi_scores[(virus_seq, reference_seq)].append(hi_value)
        virus_seqs.add(virus_seq)
        vaccine_seqs.add(reference_seq)

    virus_seqs = list(virus_seqs)
    virus_seqs.sort()
    virus_seqs_dict = {x: i for i,x in enumerate(virus_seqs)}
    vaccine_seqs = list(vaccine_seqs)
    vaccine_seqs.sort()
    vaccine_seqs_dict = {x: i for i,x in enumerate(vaccine_seqs)}
    hi_matrix = torch.zeros(len(virus_seqs), len(vaccine_seqs))
    hi_matrix_mask = torch.zeros(len(virus_seqs), len(vaccine_seqs))
    for (virus_seq, reference_seq) in seq_pairs2hi_scores:
        # print(np.mean(seq_pairs2hi_scores[(virus_seq, reference_seq)]))
        hi_matrix[virus_seqs_dict[virus_seq], vaccine_seqs_dict[reference_seq]] = np.mean(seq_pairs2hi_scores[(virus_seq, reference_seq)])
        hi_matrix_mask[virus_seqs_dict[virus_seq], vaccine_seqs_dict[reference_seq]] = 1.0
    print(hi_matrix.size(), torch.sum(hi_matrix_mask), hi_matrix.shape)
    print(hi_matrix[hi_matrix_mask.bool()].mean())
    print(hi_matrix[hi_matrix_mask.bool()].max())
    print(hi_matrix[hi_matrix_mask.bool()].min())
    

    df_pairs = pd.read_csv(args.index_pair)
    print(len(df_pairs))
    
    seq_pair_cnt = 0
    query_virus_seqs = list(set([accid2seq[x] for x in df_pairs["virus"]]))
    query_vaccine_seqs = list(set([accid2seq[x] for x in df_pairs["reference"]]))
    query_virus_seqs.sort()
    query_virus_seqs_dict = {x: i for i,x in enumerate(query_virus_seqs)}
    query_vaccine_seqs.sort()
    query_vaccine_seqs_dict = {x: i for i,x in enumerate(query_vaccine_seqs)}
    
    print("Query viruses:", len(query_virus_seqs))
    print("Viruses in HI data:", len(virus_seqs))
    query_virus_seq2virus_seqs = np.zeros((len(query_virus_seqs), len(virus_seqs)))
    print(query_virus_seq2virus_seqs.shape)
    for query_virus_seq in tqdm(query_virus_seqs):
        for virus_seq in virus_seqs:
            score = aligner.score(query_virus_seq, virus_seq)
            query_virus_seq2virus_seqs[query_virus_seqs_dict[query_virus_seq], virus_seqs_dict[virus_seq]] = score
    print(np.mean(query_virus_seq2virus_seqs), np.max(query_virus_seq2virus_seqs), np.min(query_virus_seq2virus_seqs))
    
    print("Query vaccines:", len(query_vaccine_seqs))
    print("Vaccines in HI data:", len(vaccine_seqs))
    query_vaccine_seq2vaccine_seqs = np.zeros((len(query_vaccine_seqs), len(vaccine_seqs)))
    print(query_vaccine_seq2vaccine_seqs.shape)
    for query_vaccine_seq in tqdm(query_vaccine_seqs):
        for vaccine_seq in vaccine_seqs:
            # score = 1.0
            score = aligner.score(query_vaccine_seq, vaccine_seq)
            query_vaccine_seq2vaccine_seqs[query_vaccine_seqs_dict[query_vaccine_seq], vaccine_seqs_dict[vaccine_seq]] = score
    print(np.mean(query_vaccine_seq2vaccine_seqs), np.max(query_vaccine_seq2vaccine_seqs), np.min(query_vaccine_seq2vaccine_seqs))

    query_virus_seq2virus_seqs = torch.tensor(query_virus_seq2virus_seqs)
    query_vaccine_seq2vaccine_seqs = torch.tensor(query_vaccine_seq2vaccine_seqs)
    
    # batch_size=100
    batch_size = 100 * 300 * 200 * 100 // (len(virus_seqs) * len(vaccine_seqs) * len(query_vaccine_seq2vaccine_seqs)) 
    if batch_size == 0:
        batch_size = 1
    print("batch_size", batch_size)
    # splitting the viruses, to void OOM
    ave_hi_all = []
    for i in tqdm(range(len(query_virus_seq2virus_seqs))[::batch_size]):
        # [V, V_hi] -> [V, V_hi, 1, 1]
        batched_query_viruses = query_virus_seq2virus_seqs[i:i+batch_size]
        # print("batched_query_viruses", batched_query_viruses.size())
        a = batched_query_viruses.reshape(batched_query_viruses.size(0), batched_query_viruses.size(1), 1, 1)
        a = a.expand(-1, -1, len(vaccine_seqs), len(query_vaccine_seqs))
        # print(a.size())
        # [R, R_hi] -> [R_hi, R] -> [1, 1, R_hi, R]
        # print("query_vaccine_seq2vaccine_seqs", query_vaccine_seq2vaccine_seqs.size())
        b = query_vaccine_seq2vaccine_seqs.T.reshape(1, 1, query_vaccine_seq2vaccine_seqs.size(1), query_vaccine_seq2vaccine_seqs.size(0))
        # b = np.reshape(query_vaccine_seq2vaccine_seqs.transpose(0, 1), (1, 1, query_vaccine_seq2vaccine_seqs.shape[1], query_vaccine_seq2vaccine_seqs.shape[0]))
        b = b.expand(len(batched_query_viruses), len(virus_seqs), -1, -1)
        d = a + b
        
        mask = hi_matrix_mask.view(1, hi_matrix_mask.size(0), hi_matrix_mask.size(1), 1)
        mask = mask.expand(d.size(0), -1, -1, d.size(-1))
       
        d.masked_fill_(~mask.bool(), -np.inf) # If not HI values are avaible, fill the distance as -inf
        d_flat = d.reshape(d.size(0), -1, d.size(-1))
        max_v = torch.max(d_flat, dim=1, keepdims=True)[0]
        max_indices = (d_flat == max_v)
        hi_matrix_expand = hi_matrix.view(1, hi_matrix.size(0), hi_matrix.size(1), 1).expand(d.size(0), -1, -1, d.size(-1))
        hi_matrix_expand = hi_matrix_expand.view(d.size(0), -1, d.size(-1))
        ave_hi = torch.sum(hi_matrix_expand * max_indices, dim=1) / torch.sum(max_indices, dim=1) # [V_batch, R]
        ave_hi_all.append(ave_hi)
    ave_hi_all = torch.cat(ave_hi_all, dim=0)
    ave_hi = ave_hi_all
    print(ave_hi.size())
    
    print("Searching pairs:")
    hi_value_col = []
    for virus_id, reference_id in tqdm(zip(df_pairs["virus"], df_pairs["reference"])):
        virus_seq = accid2seq[virus_id]
        reference_seq = accid2seq[reference_id]
        
        if (virus_seq, reference_seq) in seq_pairs2hi_scores:
            hi_values = seq_pairs2hi_scores[(virus_seq, reference_seq)]
            seq_pair_cnt += 1
            hi_value = sum(hi_values) / len(hi_values)
        else:
            hi_value = ave_hi[query_virus_seqs_dict[virus_seq], query_vaccine_seqs_dict[reference_seq]].item()
        
        hi_value_col.append(hi_value)
    
    print("Saving results to %s" % args.save_path)
    df_pairs["label"] = hi_value_col
    df_pairs.to_csv(args.save_path)