from Bio import SeqIO
import pandas as pd
import re, os
import numpy as np
from collections import defaultdict
import argparse

def read_fasta(path, remove_x=False):
    id2record = {}
    id2subtype = {}
    id2accid = {}
    accid2seq = {}
    id2time = {}
    for record in SeqIO.parse(path, "fasta"):
        descs = record.description.split("|")
        try:
            time = int(descs[5].split("-")[0])
            accid = descs[0]
            assert "EPI_ISL" in accid, accid
            
            id2record[descs[1].replace(" ", "_").replace("-", "_").lower()] = str(record.seq)
            id2subtype[descs[1].replace(" ", "_").replace("-", "_").lower()] = descs[2]
            id2time[descs[1].replace(" ", "_").replace("-", "_").lower()] = time
            id2accid[descs[1].replace(" ", "_").replace("-", "_").lower()] = accid
            accid2seq[accid] = str(record.seq)
        except Exception as e:
            continue
    return accid2seq, id2record, id2accid

def save(path, relative_hi, indices, pair2alignment=None):
    with open(path, "w") as fout:
        if pair2alignment is not None:
            fout.write("virus,reference,virus_seq,reference_seq,hi\n") 
        else:
            fout.write("virus,reference,hi\n") 
        for i in indices:
            if pair2alignment is not None:
                virus, ref, log_fold_hi = relative_hi[i]
                # if (ref, virus) not in pair2alignment:
                    # continue
                ref_seq, virus_seq = pair2alignment[(ref, virus)]
                fout.write("%s,%s,%s,%s,%s\n" % (virus, ref, virus_seq, ref_seq, log_fold_hi))
            else:
                virus, ref, log_fold_hi = relative_hi[i]
                fout.write("%s,%s,%s\n" % (virus, ref, log_fold_hi))

def read_alignment(path):
    pair2alignment = dict()
    with open(path) as fin:
        for line in fin:
            query,target,qaln,taln,qstart,qend,tstart,tend,mismatch = line.strip().split()
            pair2alignment[(query, target)] = (qaln, taln)
    return pair2alignment

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pairs_dir', type=str, default="/data/rsg/nlp/wenxian/esm/data/who_flu/")
    parser.add_argument('--subtype', type=str, default="a_h3n2")
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    save_path = args.subtype
    save_path = save_path + "_seed=%d" % args.seed    
    save_path = os.path.join(args.pairs_dir, save_path)

    setattr(args, "save_path", save_path)
    return args

if __name__ == "__main__":

    args = parse_args()

    np.random.seed(args.seed)

    pairs_path = os.path.join(args.pairs_dir, "%s_hi_folds.csv" % args.subtype) #  "/data/rsg/nlp/wenxian/esm/data/who_flu/" % subtype # [accid1, accid2, hi_fold_value]
    
    alignment_path = os.path.join(args.pairs_dir, "%s.m8" % args.subtype) #  "/data/rsg/nlp/wenxian/esm/data/who_flu/%s.m8" % subtype  # output alignment for msa_transformer?
    if alignment_path:
        pair2alignment = read_alignment(alignment_path)
    else:
        pair2alignment = None
    
    df = pd.read_csv(pairs_path)
    print(df)
    virus_set = list(set(df["virus"]))
    virus_set.sort()
    virus_dict = {x: i for i, x in enumerate(virus_set)}

    reference_set = list(set(df["reference"]))
    reference_set.sort()
    reference_dict = {x: i for i, x in enumerate(reference_set)}

    print("reference_set", len(reference_set), "virus_set", len(virus_set))
    
    relative_hi = list()
    for virus, ref, hi_fold in zip(df["virus"], df["reference"], df["hi"]):
        relative_hi.append((virus, ref, hi_fold))
    print(relative_hi[0])    
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # randomly split
    print("\n>>> Split pairs")
    print("# of pairs", len(relative_hi))
    index = np.random.permutation(len(relative_hi))
    train_index = index[:int(len(index) * 0.8)]
    valid_index = index[int(len(index) * 0.8):int(len(index) * 0.9)]
    test_index = index[int(len(index) * 0.9):]
    print(len(train_index), len(valid_index), len(test_index))
    print(len(train_index) + len(valid_index) + len(test_index))
    save_dir = os.path.join(args.save_path, "random_split")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save(os.path.join(save_dir, "train.csv"), relative_hi, train_index, pair2alignment=pair2alignment)
    save(os.path.join(save_dir, "valid.csv"), relative_hi, valid_index, pair2alignment=pair2alignment)
    save(os.path.join(save_dir, "test.csv"), relative_hi, test_index, pair2alignment=pair2alignment)
