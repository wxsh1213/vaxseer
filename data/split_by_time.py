from Bio import SeqIO
from collections import defaultdict
import os, sys

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_dir = input_path.split(".fasta")[0] + "_bins"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    records = SeqIO.parse(input_path, "fasta")
    time_bin_to_records = defaultdict(list)
    for record in records:
        desc = record.description.split()[1].split("|")
        desc = {x.split("=")[0]: x.split("=")[1] for x in desc}
        time_bin = desc["time_bin"]
        time_bin_to_records[time_bin].append(record)
    
    for time_bin in time_bin_to_records:
        output_path = os.path.join(output_dir, time_bin + ".fasta")
        with open(output_path, "w") as fout:
            for record in time_bin_to_records[time_bin]:
                fout.write(">%s\n%s\n\n" % (record.description, str(record.seq)))

    
    
