import pandas as pd
import numpy as np
import os, argparse
from collections import defaultdict
from Bio import SeqIO

def get_accid2seq(path):
    accid2seq = dict()
    for record in SeqIO.parse(path, "fasta"):
        for x in record.description.split("|"):
            if "EPI" in x and "EPI_ISL" not in x:
                acc_id = x.strip()
                accid2seq[acc_id] = str(record.seq)
                break
    return accid2seq

def parse_args():
    # prob_pred_path = "/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2020-04_6M/a_h3n2/human_minBinSize1000_lenQuantile0.2/test_1704_to_2004_34/lightning_logs/version_0/predictions.csv"
    # hi_pred_path = "/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2020-04/a_h1n1_and_h3n2_seed=0/random_split/predict_201704_to_202004/lightning_logs/version_0/predictions.csv"
    # hi_ids_path = "/data/rsg/nlp/wenxian/esm/data/gisaid/flu/ha_processed/2017-04_to_2020-04_9999M/a_h3n2/human_minBinSize1000_lenQuantile0.2_minCnt5.csv"
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--prob_pred_path', default=None, type=str, help="The prediction.csv output by the LM.")
    parser.add_argument('--hi_pred_path', default=None, type=str, help="The prediction.csv output by the HI predictor.")
    parser.add_argument('--hi_ids_path', default=None, type=str, help="The original .csv file for pairs.")
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--all_sequences_path', default="/data/rsg/nlp/wenxian/esm/data/gisaid/flu/ha.fasta", type=str)
    args = parser.parse_args()
    return args

# calculate the score for each candidiate vaccines, according to the prediction of dominance and HI prediction
args = parse_args()
if args.save_path is None:
    save_path = os.path.join(os.path.split(args.hi_pred_path)[0], "vaccine_score.csv") 
else:
    save_path = args.save_path

# Read accid2seq map
accid2seq = get_accid2seq(args.all_sequences_path)

if args.prob_pred_path.endswith(".csv"):
    prob = pd.read_csv(args.prob_pred_path)
    prob["prediction"] = np.exp(-prob["prediction"])
    id2freq_list = defaultdict(list)
    for src_id, pred in zip(prob["src_id"], prob["prediction"]):
        id2freq_list[src_id].append(pred)
    id2freq = {k: np.mean(v) for k, v in id2freq_list.items()}
    # assert len(prob["src_id"]) == len(set(prob["src_id"]))
    # id2freq = dict(zip(prob["src_id"], np.exp(-prob["prediction"])))
    
elif args.prob_pred_path.endswith(".fasta"):
    id2freq = {}
    for record in SeqIO.parse(args.prob_pred_path, "fasta"):
        id = record.id
        freq = float({x.split("=")[0]: x.split("=")[1] for x in record.description.split()[1].split("|")}["freq"])
        id2freq[id] = freq

sum_of_freq = sum(id2freq.values())
print("Sum of frequency in the testing viruses set:", sum_of_freq)
id2freq = {k: v / sum_of_freq for k, v in id2freq.items()} # Normalized the probability

if args.hi_ids_path:
    hi = pd.read_csv(args.hi_pred_path)
    hi_ids = pd.read_csv(args.hi_ids_path)
    hi_ids["label"] = hi["label"] # ["label"] / ["prediction"]
else:
    hi_ids = pd.read_csv(args.hi_pred_path)
# hi_ids["label"][hi_ids["label"] < 0] = 0.0

vaccine_score = defaultdict(float)
vaccine2seq = dict()
vaccine2total_freq = defaultdict(float)
for virus, ref, pred_hi in zip(hi_ids["virus"], hi_ids["reference"], hi_ids["label"]):
    if not np.isnan(pred_hi):
        vaccine_score[ref] += id2freq.get(virus, 0.0) * pred_hi
        vaccine2total_freq[ref] += id2freq.get(virus, 0.0)
        # norm += id2freq.get(virus, 0.0)
    else:
        vaccine_score[ref] += 0.0
    vaccine2seq[ref] = accid2seq[ref]

print("Read vaccine #:", len(vaccine_score))

ave_vaccine_score = sum(vaccine_score.values()) / len(vaccine_score)
print("Average vaccine score", ave_vaccine_score)

print("Saving to %s" % save_path)
with open(save_path, "w") as fout:
    fout.write("reference,reference_seq,score\n")
    for vaccine, score in vaccine_score.items():
        if vaccine2total_freq[vaccine] > 0:
            fout.write("%s,%s,%g\n" % (vaccine, vaccine2seq[vaccine], score / vaccine2total_freq[vaccine]))
        else:
            # print("Could find the vaccine score for %s, set it as the average score." % vaccine)
            print("Warning: could find the vaccine score for %s, set it as nan." % vaccine)
            fout.write("%s,%s,\n" % (vaccine, vaccine2seq[vaccine],))
