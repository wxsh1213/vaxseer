from Bio import SeqIO
from collections import defaultdict
import os, sys

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_dir = input_path.split(".fasta")[0] + "_locations"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    records = SeqIO.parse(input_path, "fasta")
    location_to_records = defaultdict(list)
    for record in records:
        desc = record.description.split()[1].split("|")
        desc = {x.split("=")[0]: x.split("=")[1] for x in desc}
        location = desc["location"]
        location_to_records[location].append(record)
    
    for location in location_to_records:
        output_path = os.path.join(output_dir, location + ".fasta")
        if not os.path.exists(os.path.split(output_path)[0]):
            os.makedirs(os.path.split(output_path)[0])
        with open(output_path, "w") as fout:
            for record in location_to_records[location]:
                fout.write(">%s\n%s\n\n" % (record.description, str(record.seq)))

    
    
