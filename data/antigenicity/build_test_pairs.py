import sys
# from tools.io import read_fasta, read_m8
# from tools.meta import parse_description
import argparse
from process_fasta import read_fasta
from collections import defaultdict

def parse_description(description):
    desc = description.split()[1]
    descs = desc.split("|")
    return {x.split("=")[0]: x.split("=")[1] for x in descs}


def read_m8(path, remove_self=True):
    query2targets = defaultdict(list)
    with open(path) as f:
        for line in f:
            qid, tid, qaln, taln, qstart, qend, tstart, tend, mmnum = line.strip().split()
            if remove_self and qid == tid:
                continue
            query2targets[qid].append((qid, tid, qaln, taln, qstart, qend, tstart, tend, mmnum))
    return query2targets

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--alignment_path', default=None, type=str)
    parser.add_argument('--virus_fasta_path', default=None, type=str)
    parser.add_argument('--vaccine_fasta_path', default=None, type=str)
    parser.add_argument('--save_path', default=None, type=str)
      
    args = parser.parse_args()
    return args

args = parse_args()

min_virus_freq = 0
min_vaccine_freq = 0

virus_seqs = read_fasta(args.virus_fasta_path)
vaccine_seqs = read_fasta(args.vaccine_fasta_path)
alignments = read_m8(args.alignment_path)

# filter
def filter_data_by_count(records, min_freq):
    new_records = []
    for idd, seq, desc in records:
        desc_dict = parse_description(desc)
        cnt = float(desc_dict["freq"]) * float(desc_dict["bin_size"])
        if cnt < min_freq:
            continue
        new_records.append((idd, seq, desc))
    return new_records

if min_virus_freq > 0:
    virus_seqs = filter_data_by_count(virus_seqs, min_freq=min_virus_freq)
if min_vaccine_freq > 0:
    vaccine_seqs = filter_data_by_count(vaccine_seqs, min_freq=min_vaccine_freq)
print(len(virus_seqs), len(vaccine_seqs))

virus_ids = set([x[0] for x in virus_seqs])
vaccine_ids = set([x[0] for x in vaccine_seqs])

pairs2aligns = {}
for query in alignments:
    for qid, tid, qaln, taln, qstart, qend, tstart, tend, mmnum in alignments[query]:
        if qid in virus_ids and tid in vaccine_ids:
            pairs2aligns[(qid, tid)] = (qaln, taln)


num_pairs = 0
num_coundn_find_alignment = 0
with open(args.save_path, "w") as fout:
    fout.write("virus,reference,virus_seq,reference_seq,hi\n")
    for qid, qseq, qdesc in virus_seqs:
        for tid, tseq, tdesc in vaccine_seqs:
            if (qid, tid) in pairs2aligns:
                qaln, taln = pairs2aligns[(qid, tid)]
                assert len(qaln) == len(taln)
                fout.write("%s,%s,%s,%s,0\n" % (qid, tid, qaln, taln))
                num_pairs += 1
            else:
                num_coundn_find_alignment += 1
print("num_coundn_find_alignment", num_coundn_find_alignment)
print("num_pairs", num_pairs)
