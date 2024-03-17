from Bio import SeqIO, Phylo
import logging, string, pickle
import csv

def read_meta(path):
    meta_data = {}
    with open(path) as csvfile:
        spamreader = csv.reader(csvfile)
        for i, row in enumerate(spamreader):
            if i == 0:
                headline = row
                isl_index = headline.index("Isolate_Id")
            else:
                isl_id = row[isl_index]
                meta_data[isl_id] = dict(zip(headline, row))
    return meta_data

def read_fasta(file, quiet=False):
    records = []
    try:
        with open(file) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # day = int(record.description.split()[1])
                if "|" in record.id:
                    ids = record.id.split("|")[0]
                else:
                    ids = record.id
                records.append((ids, str(record.seq), record.description))
        if not quiet:
            logging.info("Read %d test samples from %s" % (len(records), file))
    except Exception as e:
        raise ValueError("Fail to read from file: %s" % file)
    return records

def read_m8(path):
    data_list = []
    with open(path) as f:
        for line in f:
            qid, tid, qseq, tseq, qstart, qend, tstart, tend = line.strip().split()
            data_list.append({
                "qid": qid, "tid": tid,
                "qseq": qseq, "tseq": tseq, 
                "qstart": int(qstart), "qend": int(qend),
                "tstart": int(tstart), "tend": int(tend),
            })
    return data_list

def read_newick_tree(path):
    tree = Phylo.read(path, "newick")
    print("Read tree from %s" % path)
    return tree

def read_tree(path):
    try:
        tree = pickle.load(open(path, "rb"))
    except Exception as e:
        tree = read_newick_tree(path)
        tree = Phylo.to_networkx(tree)
    finally:
        return tree
        
def remove_insertions(sequence: str):
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)

    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str):
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]