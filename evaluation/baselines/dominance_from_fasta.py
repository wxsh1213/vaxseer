from Bio import SeqIO
import numpy as np
import pandas as pd
import os, argparse, subprocess
from collections import defaultdict

def read_seq2freq(paths):
    seq2count = defaultdict(int)
    for path in paths:
        for record in SeqIO.parse(path, "fasta"):
            seq = str(record.seq)
            freq = float({x.split("=")[0]: x.split("=")[1] for x in record.description.split()[1].split("|")}["freq"])
            bin_size = float({x.split("=")[0]: x.split("=")[1] for x in record.description.split()[1].split("|")}["bin_size"])
            seq2count[seq] += round(freq * bin_size)
    total_count = sum(seq2count.values())
    seq2freq = {k: v / total_count for k, v in seq2count.items()}
    return seq2freq

def read_fasta(path):
    records = []
    for record in SeqIO.parse(path, "fasta"):
        records.append((record.id, str(record.seq), record.description))
    return records

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_fasta_paths', default=None, type=str, nargs="+")
    parser.add_argument('--target_fasta_path', default=None, type=str)
    parser.add_argument('--save_path', default=None, type=str)
    args = parser.parse_args()
    return args

args = parse_args()

source_seq2freq = read_seq2freq(args.source_fasta_paths)
targets = read_fasta(args.target_fasta_path)

all_freqs = 0.0
src_ids, predictions = [], []
for target_id, target_seq, target_desc in targets:
    freq = - np.log(source_seq2freq.get(target_seq, 0.0))
    src_ids.append(target_id)
    predictions.append(freq)
    all_freqs += source_seq2freq.get(target_seq, 0.0)
df = pd.DataFrame(data={"src_id": src_ids, "prediction": predictions})
df.to_csv(args.save_path)

print("all_freqs:", all_freqs)