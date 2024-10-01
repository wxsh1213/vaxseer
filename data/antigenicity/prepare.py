from Bio import SeqIO
import pandas as pd
import re, os
import numpy as np
from collections import defaultdict
import argparse

def seach_id(id2record, query):
    matches = []
    for idd in id2record:
        if query in idd:
            matches.append(idd)
    return matches

def read_fasta(path, max_date = "9999-12-31"):
    max_year, max_month = int(max_date.split("-")[0]), int(max_date.split("-")[1])
    
    min_seq_length = 553
    print("Min seq length", min_seq_length)

    id2records = defaultdict(set)
    seq2accid = defaultdict(list)
    id2accids = defaultdict(set)
    accid2seq = {}
    accid2id = {}
    accid2date = {}
    for record in SeqIO.parse(path, "fasta"):
        descs = record.description.split("|")
        try:
            if len(str(record.seq)) < min_seq_length:
                continue

            ha_id = None
            date = None
            for x in descs:
                if "EPI" in x and "EPI_ISL" not in x:
                    ha_id = x
                
                matchObj = re.match(r'([0-9]*)-([0-9]*)-*([0-9]*)', x, re.M|re.I)
                if matchObj is not None:
                    date = x

            assert "EPI" in ha_id, descs
            if date is None:
                continue

            year, month = int(date.split("-")[0]), int(date.split("-")[1])
            if year > max_year or (year == max_year and month >= max_month):
                continue
            
            id2records[descs[1].replace(" ", "_").replace("-", "_").lower()].add(str(record.seq))
            id2accids[descs[1].replace(" ", "_").replace("-", "_").lower()].add(ha_id)
            
            accid2id[ha_id] = descs[1].replace(" ", "_").replace("-", "_").lower()
            
            accid2seq[ha_id] = str(record.seq)
            accid2date[ha_id] = date
            seq2accid[str(record.seq)].append(ha_id)
        except Exception as e:
            print(e)
            continue

    return accid2seq, id2records, id2accids, seq2accid, accid2id, accid2date

def alias_name(all_ids, query_id):
    if len(query_id.split("/")[-1]) == 2:
        query_id = "/".join(query_id.split("/")[:-1]) + "/20" + query_id.split("/")[-1]

    if "(" in query_id:
        query_id_short = re.findall(r"[(](.*?)[)]", query_id)[0]
        if len(query_id_short.split("/")[-1]) == 2:
            if int(query_id_short.split("/")[-1][0]) <= 2:
                query_id_short = "/".join(query_id_short.split("/")[:-1]) + "/20" + query_id_short.split("/")[-1]
            else:
                query_id_short = "/".join(query_id_short.split("/")[:-1]) + "/19" + query_id_short.split("/")[-1]

        prefix = query_id.split("(")[0]
        matches = seach_id(all_ids, query_id_short)
        new_name = None
        for match in matches:
            if prefix in match:
                new_name = match
        if new_name is None:
            new_name = query_id_short
        query_id = new_name
    return query_id

def save(path, relative_hi, indices):
    with open(path, "w") as fout:
        fout.write("virus,reference,hi\n") 
        for i in indices:
            virus, ref, log_fold_hi = relative_hi[i]
            fout.write("%s,%s,%s\n" % (virus, ref, log_fold_hi))


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pairs_path', default="", type=str)
    parser.add_argument('--sequences_path', default="", type=str)
    parser.add_argument('--max_date_exclude', default=None, type=str)
    parser.add_argument('--output_root_dir', default=".", type=str)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_root_dir, "before_%s" % (args.max_date_exclude))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sub_type_name = "_".join(os.path.split(args.pairs_path)[-1].split("_")[:2])
    sequences_output_path = os.path.join(output_dir, "%s.fasta" % sub_type_name)
    setattr(args, "sequences_output_path", sequences_output_path)
    return args


if __name__ == "__main__":
    args = parse_args()

    accid2seq, id2seqs, strain_name2accids, seq2acc_ids, accid2id, accid2date = read_fasta(args.sequences_path, max_date=args.max_date_exclude)
    print("Max date (included)", max(accid2date.values()))

    gisaid_ids = set(id2seqs.keys())
    
    df = pd.read_csv(args.pairs_path)
    virus_set = list(set(df["virus"]))
    reference_set = set(df["reference"])
    print(len(reference_set), len(virus_set))
    fail_count = 0
    fail_count_2 = 0

    ori_virus_name_to_accids = {}
    for virus in virus_set:
        alias_id = alias_name(gisaid_ids, virus)
        if alias_id in strain_name2accids:
            ori_virus_name_to_accids[virus] = strain_name2accids[alias_id]
    
    ori_ref_name_to_accids = {}
    for virus in reference_set:
        alias_id = alias_name(gisaid_ids, virus)
        if alias_id in strain_name2accids:
            ori_ref_name_to_accids[virus] = strain_name2accids[alias_id]
        

    df["hi"] = np.log2(df["hi"])
    ref2ref_hi = dict()
    for ref, virus, hi in zip(df["reference"], df["virus"], df["hi"]):
        if virus == ref:
            ref2ref_hi[ref] = hi

    normalized_his = []
    bool_mask = []
    for ref, virus, hi in zip(df["reference"], df["virus"], df["hi"]):
        if ref in ref2ref_hi:
            hi_new = ref2ref_hi[ref] - hi
            hi_new = max(hi_new, 0.0)
            normalized_his.append(hi_new)
            bool_mask.append(True)
        else:
            bool_mask.append(False)
    bool_mask = np.asarray(bool_mask, dtype=np.bool8)
    df = df[bool_mask]
    df["hi"] = normalized_his


    virus_set = set()
    virus_seq_set = set()
    for virus in ori_virus_name_to_accids:
        virus_set.update(ori_virus_name_to_accids[virus])
        virus_seq_set.update([accid2seq[idd] for idd in ori_virus_name_to_accids[virus]])
    print(len(virus_set), len(virus_seq_set))
    
    reference_set = set()
    reference_seq_set = set()
    seq2ori_ref_name = defaultdict(list)
    for ref in ori_ref_name_to_accids:
        reference_set.update(ori_ref_name_to_accids[ref])
        reference_seq_set.update([accid2seq[idd] for idd in ori_ref_name_to_accids[ref]])
        for seq in [accid2seq[idd] for idd in ori_ref_name_to_accids[ref]]:
            seq2ori_ref_name[seq].append(ref)
    
    all_seq_set = virus_seq_set | reference_seq_set
    print(len(reference_set), len(virus_set), len(all_seq_set))

    print(">>> Saving all sequences to %s" % args.sequences_output_path)
    with open(args.sequences_output_path, "w") as fout:
        for seq in all_seq_set:
            fout.write(">%s\n%s\n\n" % (seq2acc_ids[seq][0], seq))
    
    print(">>> Saving virus sequences to %s" % (args.sequences_output_path.split(".fasta")[0] + "_virus.fasta"))
    with open(args.sequences_output_path.split(".fasta")[0] + "_virus.fasta", "w") as fout:
        for seq in virus_seq_set:
            fout.write(">%s\n%s\n\n" % (seq2acc_ids[seq][0], seq))
    
    print(">>> Saving vaccine sequences to %s" % (args.sequences_output_path.split(".fasta")[0] + "_vaccine.fasta"))
    with open(args.sequences_output_path.split(".fasta")[0] + "_vaccine.fasta", "w") as fout:
        for seq in reference_seq_set:
            fout.write(">%s\n%s\n\n" % (seq2acc_ids[seq][0], seq))
    
    ref_hi = defaultdict(list)
    new_pairs = defaultdict(list)
    for virus, ref, hi in zip(df["virus"], df["reference"], df["hi"]):
        if virus in ori_virus_name_to_accids:
            if ref in ori_ref_name_to_accids:
                for virus_accid in ori_virus_name_to_accids[virus]:
                    for ref_accid in ori_ref_name_to_accids[ref]:
                        virus_seq = accid2seq[virus_accid]
                        ref_seq = accid2seq[ref_accid]
                        new_pairs[(virus_seq, ref_seq)].append(hi) # .append((virus_accid, ref_accid, hi))
                        
                        if virus_seq == ref_seq: # TODO: HERE!!!!
                            ref_hi[ref_seq].append(hi)


    relative_hi = list()
    for virus, ref in new_pairs:
        log_his = new_pairs[(virus, ref)]
        loghi = np.mean(log_his)
        
        virus_accid = seq2acc_ids[virus][0]
        ref_accid = seq2acc_ids[ref][0]

        ref_ori_names = list(set(seq2ori_ref_name[ref]))
        ref_ori_names.sort()

        relative_hi.append((virus_accid, ref_accid, loghi, ref_ori_names))
    
    print(relative_hi[0])
    
    print("# of pairs", len(relative_hi))
    hi_value_output_path = os.path.join(args.sequences_output_path.split(".fasta")[0] + "_hi_folds.csv")
    print("Saving HI values to %s" % hi_value_output_path)
    with open(hi_value_output_path, "w") as fout:
        fout.write("virus,reference,hi\n") 
        for virus, ref, log_fold_hi, ref_ori_names in relative_hi:
            fout.write("%s,%s,%s\n" % (virus, ref,log_fold_hi))


