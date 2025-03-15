import sys, re, os
from collections import defaultdict
from tqdm import tqdm

def parse_aa_substitution(aa_subs, offset=0):
    aa_subs_list = []
    for aa_sub in list(aa_subs):
        matchObj = re.match( r'(\D*?)(\d+)(\D*)', aa_sub, re.M|re.I)
        if matchObj is None:
            continue
        wt = matchObj.group(1)
        location = int(matchObj.group(2)) - offset
        mt = matchObj.group(3)
        aa_subs_list.append((wt, location, mt))
    return aa_subs_list

def extract_aa_substitutions(meta):
    aa_substitutions = meta["AA Substitutions"]
    single_aa_mutants = set()
    for x in aa_substitutions[1:-1].split(","):
        if "Spike_" not in x:
            continue
        m = x.split("Spike_")[1]
        single_aa_mutants.add(m)
    return single_aa_mutants

def read_meta_data(path, return_list=False):
    data = []
    id_to_data = {}
    with open(path) as f:
        head_line = f.readline().strip().split("\t")
        print(head_line)
        for line in f:
            data.append(dict(zip(head_line, line.strip().split("\t"))))
    if return_list:
        return data
    id_to_data = {x["Accession ID"]: x for x in data}
    print("Read %d meta data from %s" % (len(id_to_data), path))
    return id_to_data

def get_sequence_from_aa_substitutions(ref_seq, aa_subs, remove_gap=True, early_stop=True):
    new_seq = list(ref_seq)
    for wt, loc, mt in aa_subs:
        if wt != "ins":
            assert ref_seq[loc] == wt, "Wide type mismatch: %s v.s. %s at %d" % (ref_seq[loc], wt, loc)
        else:
            continue

        if mt != "del" and mt != "stop": # substitution
            new_seq[loc] = mt
        elif mt == "del": # deletion
            new_seq[loc] = "-"
        elif mt == "stop": # replace by a stop
            new_seq[loc] = "*"
    for wt, loc, mt in aa_subs:
        if wt == "ins": # insertion
            new_seq[loc] = new_seq[loc] + mt

    if early_stop:
        for i, c in enumerate(new_seq):
            if c == "*":
                i = i - 1
                break
        new_seq = new_seq[:i+1]

    if remove_gap:
        return "".join([x for x in new_seq if x != "-"])
    else:
        return "".join(new_seq)
    
ref_fullspike_seq= "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT*"
rbd_start, rbd_end = 319, 541
ref_rbd_seq = ref_fullspike_seq[rbd_start-1:rbd_end] # str(next(SeqIO.parse("/data/rsg/nlp/wenxian/esm/data/cov_spike/reference_rbd.fasta", "fasta")).seq)


def save(saving_path, results):
    accid_with_multiple_sequences = 0
    with open(saving_path, "w") as fout:
        for acc_id in results:
            if len(set(results[acc_id])) == 1:
                fout.write(">%s\n%s\n\n" % (acc_id, results[acc_id][0]))
            else:
                accid_with_multiple_sequences += 1
                _seqs = list(set(results[acc_id]))
                _seqs.sort()
                for subaccid in range(len(_seqs)):
                    fout.write(">%s_%d\n%s\n\n" % (acc_id, subaccid, _seqs[subaccid]))
    
    print("accid_with_multiple_sequences:", accid_with_multiple_sequences)


if __name__ == "__main__":
    meta_data_path = sys.argv[1]
    
    time_stamp = meta_data_path.split("cov_meta_")[1].split(".tsv")[0]
    print("time_stamp", time_stamp)

    rbd_saving_path = os.path.join(os.path.split(meta_data_path)[0], "rbd_from_meta_%s.fasta" % time_stamp) 

    meta_data = read_meta_data(meta_data_path, return_list=True)

    rbd_results = defaultdict(list)
    error_count = 0
    sucess_count = 0

    for item in tqdm(meta_data):
        accession_id = item['Accession ID']
        aa_subs = item['AA Substitutions']

        try:
            aa_subs = extract_aa_substitutions(item)

            rbd_aa_subs_parsed = parse_aa_substitution(aa_subs, offset=rbd_start)
            rbd_aa_subs_parsed = [(wt, loc, mt) for wt, loc, mt in rbd_aa_subs_parsed if loc >= 0 and loc <= rbd_end-rbd_start]
            rbd_seq = get_sequence_from_aa_substitutions(ref_rbd_seq, rbd_aa_subs_parsed, remove_gap=True, early_stop=True)
            rbd_results[accession_id].append(rbd_seq)

            sucess_count += 1

        except Exception as e:
            print(e)
            error_count += 1

    print("error_count", error_count)
    print("sucess_count", sucess_count)
    print(len(rbd_results))

    save(rbd_saving_path, rbd_results)

