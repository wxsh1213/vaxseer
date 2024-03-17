import argparse
from Bio import SeqIO
import pandas as pd
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluating the generation performance')
    parser.add_argument('--hi_form_path', default="/data/rsg/nlp/wenxian/esm/data/who_flu/a_h3n2_hi_folds.csv", type=str)
    parser.add_argument('--sequence_file', default="/data/rsg/nlp/wenxian/esm/data/gisaid/flu/ha.fasta", type=str)
    parser.add_argument('--index_pair', default="", type=str)
    parser.add_argument('--save_path', default="", type=str)
    parser.add_argument('--no_imputation', action="store_true")
    parser.add_argument('--impute_by_vaccine_avg', action="store_true", help="Impute the missing vaccine-virus pair by the average HI values for the vaccine.")
    parser.add_argument('--impute_by_vaccine_virus_clade_avg', action="store_true")
    args = parser.parse_args()
    return args

def read_fasta(path):
    id2record = {}
    id2subtype = {}
    id2accid = {}
    accid2seq = {}
    id2time = {}
    seq2strain_name = defaultdict(set)
    for record in SeqIO.parse(path, "fasta"):
        descs = record.description.split("|")
        try:
            time = int(descs[5].split("-")[0])
            
            accid = None
            for x in descs:
                if "EPI" in x and "EPI_ISL" not in x:
                    accid = x
            
            # accid = descs[0]
            # assert "EPI_ISL" in accid, accid

            assert accid is not None

            
            # id2record[descs[1].replace(" ", "_").replace("-", "_").lower()] = str(record.seq)
            # id2subtype[descs[1].replace(" ", "_").replace("-", "_").lower()] = descs[2]
            # id2time[descs[1].replace(" ", "_").replace("-", "_").lower()] = time
            # id2accid[descs[1].replace(" ", "_").replace("-", "_").lower()] = accid
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
    clade_pairs2hi_scores = defaultdict(list)
    vaccine_virus_clade2hi_scores = defaultdict(list)
    vaccine2hi_scores = defaultdict(list)
    all_hi_scores = list()
    df_hi = pd.read_csv(args.hi_form_path)

    for virus_id, reference_id, hi_value in zip(df_hi["virus"], df_hi["reference"], df_hi["hi"]):
        virus_seq = accid2seq[virus_id]
        reference_seq = accid2seq[reference_id]
        seq_pairs2hi_scores[(virus_seq, reference_seq)].append(hi_value)
        all_hi_scores.append(hi_value)

    ave_hi_score = sum(all_hi_scores)/ len(all_hi_scores)
    print(ave_hi_score)

    df_pairs = pd.read_csv(args.index_pair)
    hi_value_col = []
    seq_pair_cnt = 0

    for virus_id, reference_id in zip(df_pairs["virus"], df_pairs["reference"]):
        virus_seq = accid2seq[virus_id]
        reference_seq = accid2seq[reference_id]
        
        if (virus_seq, reference_seq) in seq_pairs2hi_scores:
            hi_values = seq_pairs2hi_scores[(virus_seq, reference_seq)]
            seq_pair_cnt += 1
            hi_value = sum(hi_values) / len(hi_values)
        else:
            hi_value = ave_hi_score
        
        hi_value_col.append(hi_value)
    
    print("#Find seq pair", seq_pair_cnt)
    df_pairs["label"] = hi_value_col
    df_pairs.to_csv(args.save_path)
    print(len(df_pairs))