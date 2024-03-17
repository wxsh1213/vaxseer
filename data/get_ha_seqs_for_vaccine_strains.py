import pandas as pd
import os, subprocess
from collections import defaultdict
from Bio import SeqIO
# from tools.io import write_fasta
import argparse


def write_fasta(records, file):
    with open(file, "w") as handle:
        for rid, seq, desc in records:
            if rid in desc:
                handle.write(">%s\n%s\n\n" % (desc, seq))
            else:
                handle.write(">%s %s\n%s\n\n" % (rid, desc, seq))
    return records

parser = argparse.ArgumentParser(description='')
parser.add_argument('--vaccine_path', default="recommended_vaccines_from_gisaid.csv", type=str)
parser.add_argument('--sequences_path', default="", type=str)
parser.add_argument('--saving_dir', default="recommended_vaccines_from_gisaid_ha", type=str)

args = parser.parse_args()

# read HA sequences from GISAID
records = SeqIO.parse(args.sequences_path, "fasta")
strain2records = defaultdict(list)
for x in records:
    description = x.description
    strain_name = description.split("|")[1]
    strain2records[strain_name].append(x)

if not os.path.exists(args.saving_dir):
    os.makedirs(args.saving_dir)

df = pd.read_csv(args.vaccine_path)

vaccines = defaultdict(list)
for season, v_h3n2, v_h1h1 in zip(df["Season"], df["A/H3N2"], df["A/H1N1"]):
    vaccines[(season, "a_h3n2")].append(v_h3n2.strip().replace(" ", "_"))
    vaccines[(season, "a_h1n1")].append(v_h1h1.strip().replace(" ", "_"))

for season, subtype in vaccines:
    data = []
    for strain_name in vaccines[(season, subtype)]:
        if len(strain2records[strain_name]) == 0:
            print("No records are found for strain %s:" % strain_name)
        for record in strain2records[strain_name]:
            epi_id = None
            for ss in record.description.split("|"):
                if "EPI" in ss and "ISL" not in ss:
                    epi_id = ss
            if epi_id is None:
                continue
            data.append((epi_id, str(record.seq), "season=%s|strain_name=%s" % (season.replace(" ", "_"), strain_name)))
    save_path = os.path.join(args.saving_dir, season.replace(" ", "_") + "_%s.fasta" % subtype.lower().replace("/", "_"))
    write_fasta(data, save_path)


