from Bio import SeqIO
import argparse, os, pickle, re, csv
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import datetime, json
from Bio import SeqIO
from datetime import datetime, date

def filter_passage(x, remove_nan=False, only_keep_original=False):
    x = x.lower()

    if only_keep_original:
        if "original" in x:
            return True
        if "clinical" in x:
            return True
        if "ori" in x:
            return True
        if "cs" in x:
            return True
        if "direct" in x:
            return True
        if "human" in x:
            return True
        if "unpassaged" in x:
            return True
        return False
    # print(remove_nan)
    # exit()
    # if not isinstance(x, str):
    #     print(x)
    #     print(isinstance(x, float) and np.isnan(x))
    #     if remove_nan:
    #         if isinstance(x, float) and np.isnan(x): # Also remove the nan (not available info)
    #             return False
        
    #     return True

    
    if "mdck" in x:
        return False
    if "e1" in x or "e2" in x or "e3" in x or "egg" in x:
        return False
    if "pmk" in x:
        return False
    if "c1" in x or "c2" in x or "c3" in x:
        return False
    if "s1" in x or "s2" in x or "s3" in x:
        return False
    if "p1" in x or "p2" in x or "p3" in x or "p0" in x:
        return False

    if remove_nan:
        if not x: # empty
            return False
        
    # if x in ("E1", "C1", "P1", "S1")
    
    return True

def read_meta(path, key_field="Isolate_Id", delimiter=","):
    meta_data = {}
    with open(path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delimiter)
        for i, row in enumerate(spamreader):
            if i == 0:
                headline = row
            else:
                row_dict = dict(zip(headline, row))
                key_value = row_dict[key_field]
                if "Segment_Id" in key_field:
                    if not key_value:
                        continue
                    key_value = key_value.split("|")[0]
                meta_data[key_value] = row_dict
    return meta_data

def validate(seq, alphabet='protein'):
    alphabets = {'dna': re.compile('^[acgtnurykmswbdhvnx]*$', re.I), 
             'protein': re.compile('^([a-z]|-|\*)*$', re.I)}

    assert alphabet in alphabets, "Unrecognize molecule name: %s" % alphabet

    if alphabets[alphabet].match(seq) is not None:
         return True
    else:
         return False

def read_fasta(file, quiet=False, ignore_error=False, molecule="protein"):
    records = []
    unrecognized_seqs_num = 0
    with open(file) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            # day = int(record.description.split()[1])
            if ignore_error:
                records.append((record.id, str(record.seq), record.description))
            else:
                if validate(str(record.seq).lower(), molecule):
                    records.append((record.id, str(record.seq), record.description))
                else:
                    if not quiet:
                        print("Unrecognized sequence for %s: %s" %(record.description, str(record.seq)) )
                        # print(recorddescription)
                    unrecognized_seqs_num += 1
    if not quiet:
        print("Read %d test samples from %s (fail count %d)" % (len(records), file, unrecognized_seqs_num))
    return records

def date_to_day(input_date, start_day = "2019-12-30"):
    try:
        return int((date(*tuple(map(int, input_date.split("-")))) - date(*tuple(map(int, start_day.split("-"))))).days) 
    except Exception as e:
        return None

def year_fraction(input_date):
    if len(input_date.split("-")) == 3:
        d = date(*tuple(map(int, input_date.split("-"))))
        start = date(d.year, 1, 1).toordinal()
        year_length = date(d.year+1, 1, 1).toordinal() - start
        return d.year + float(d.toordinal() - start) / year_length
    else:
        return None

def date_to_month(input_date, start_day = "0000-01"):
    try:
        year, month = input_date.split("-")[:2]
        year, month = int(year), int(month)
        year_base, month_base = start_day.split("-")[:2]
        year_base, month_base = int(year_base), int(month_base)
        diff = (year - year_base) * 12 + (month - month_base)
        return diff
    except Exception as e:
        return None

def date_to_year(input_date, start_day = "0000"):
    try:
        year = int(input_date.split("-")[0])
        year_base = int(start_day.split("-")[0])
        diff = year - year_base
        return diff
    except Exception as e:
        return None

def check_flu_subtype(subtype, target_subtype):
    subtype = subtype.lower()
    subtype = subtype.replace(" ", "").replace("/", "_") # A / H1N1 -> a_h1n1
    target_subtype = target_subtype.lower()

    matchObj = re.match( r'([a-z])_*h*([0-9]*)n*([0-9]*)', subtype, re.M|re.I)
    abc = matchObj.group(1) # a,b...
    htype = matchObj.group(2)
    ntype = matchObj.group(3)

    matchObj = re.match( r'([a-z])_*h*([0-9]*)n*([0-9]*)', target_subtype, re.M|re.I)
    if matchObj:
        abc_target = matchObj.group(1) # a,b...
        htype_target = matchObj.group(2) # a_h1,...
        ntype_target = matchObj.group(3)

        if not htype_target and not ntype_target: # a, b, c, d...
            if abc != abc_target:
                return False
        elif htype_target and not ntype_target: # a_h5
            if abc != abc_target or htype != htype_target:
                return False
        elif ntype_target and not htype_target:  # a_n5
            if abc != abc_target or ntype != ntype_target:
                return False
        else: # a_h3n2
            if abc != abc_target or ntype != ntype_target or htype != htype_target:
                return False
    return True
    
meta_data_acc_id_field = {
    "flu": "HA Segment_Id",
    "cov": "Accession ID"
}

meta_data_collection_date_field = {
    "flu": "Collection_Date",
    "cov": "Collection date"
}

START_DATE = {
    "flu": "1900-01-01",
    "cov": "2019-12-01"
}

def customize_save_path(args):
    save_dir = os.path.join(args.save_dir, "%s_to_%s_%d%s" % (args.start_date, args.end_date, args.time_interval, args.split_by[0].upper()))
    
    if args.subtype:
        save_dir = os.path.join(save_dir, "%s" % args.subtype.lower().replace("/", "_").replace(" ", ""))
    else:
        save_dir = os.path.join(save_dir, "all")
    
    if args.lineage:
        save_dir = save_dir + "_%s" % args.lineage # like B_yamagata
    
    if args.continent is not None:
        save_dir = os.path.join(save_dir, "continents")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, "")
    if args.host:
        save_path += "%s" % args.host
    else:
        save_path += "allhost"
    
    if args.remove_min_size > 0:
        save_path += "_minBinSize%d" % args.remove_min_size

    if args.min_seq_length >= 0:
        save_path += "_minLen%g" % args.min_seq_length
    else:
        if args.min_seq_length_quantile > 0:
            save_path += "_minlenQuantile%g" % args.min_seq_length_quantile
        
    if args.max_seq_length >= 0:
        save_path += "_maxLen%g" % args.max_seq_length
    else:
        if args.max_seq_length_quantile < 1.0:
            save_path += "_maxlenQuantile%g" % args.max_seq_length_quantile 
    
    if args.min_count > 0:
        save_path += "_minCnt%g" % args.min_count
    
    if args.continent is not None:
        save_path += "_%s" % args.continent
    
    if args.country is not None:
        save_path += "_%s" % args.country

    if len(args.identity_keys) > 0:
        save_path = save_path + "_" + "_".join(args.identity_keys)
        if "location" in args.identity_keys and args.separate_region is not None:
            save_path += "_region%s" % args.separate_region
    
    if args.filter_passage:
        save_path += "_filter_passage"
        if args.filter_passage_remove_nan:
            save_path += "_remove_nan"
        if args.filter_passage_original_only:
            save_path += "_ori_only"

    save_ids_path = save_path + ".ids"
    save_path = save_path + ".fasta"
    return save_path, save_ids_path

def get_max_collection_date(meta_data, collection_date_field="Collection_Date"):
    collection_dates = [meta_data[x][collection_date_field] for x in meta_data]
    collection_dates = [x for x in collection_dates if len(x.split("-")) == 3]
    collection_dates.sort()
    return tuple(map(int, collection_dates[-1].split("-")))

def build_id2seq_flu(sequences):
    id2seq = {}
    id2desc = {}
    for record in sequences:
        desc = [y for x in record[-1].split() for y in x.split("|")]
        ha_id = None
        for x in desc:
            if "EPI" in x and "EPI_ISL_" not in x:
                ha_id = x
        # print(ha_id)
        if ha_id is not None:
            id2seq[ha_id] = record[1]
            id2desc[ha_id] = record[-1]
    return id2seq, id2desc

def build_id2seq_cov(sequences):
    id2seq = {}
    id2desc = {}
    for record in sequences:
        if record[0][:8] == "EPI_ISL_":
            id2seq[record[0]] = record[1]
            id2desc[record[0]] = record[-1]
        else:
            print("Warning: Unrecognize accession id: %s" % record[0])
    return id2seq, id2desc

def get_location(meta_data, location_level, continent_to_countries=None):
    location = meta_data["Location"]
    if not location:
        return None
    location = tuple(x.strip().lower().replace(" ", "_") for x in location.split("/")[:location_level])
    if location_level is not None and len(location) < location_level:
        return None
    for x in location:
        if len(x) == 0:
            return None
    if len(location) >= 1:
        if continent_to_countries is not None and location[0] not in continent_to_countries:
            return None
    if len(location) >= 2:
        if continent_to_countries is not None and location[1] not in continent_to_countries[location[0]]:
            return None
    return location

regions_level=("continent", "country", "state", "city")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--antigen', default='flu', type=str, choices=["flu", "cov"])
    parser.add_argument('--meta_data_path', default="", type=str)
    parser.add_argument('--sequences_path', default="", type=str)
    parser.add_argument('--save_dir', default="", type=str)
    parser.add_argument('--time_interval', default=2, type=int)
    parser.add_argument('--start_date', default=None, type=str)
    parser.add_argument('--end_date', default=None, type=str)
    parser.add_argument('--remove_min_size', default=0, type=int, help="If the number of samples in each time slot less than the threshold, remove this time slot.")
    parser.add_argument('--subtype', default=None, type=str)
    parser.add_argument('--host', default="human", type=str)
    # parser.add_argument('--seq_length_quantile', default=0.2, type=float)
    parser.add_argument('--min_seq_length_quantile', default=0.2, type=float)
    parser.add_argument('--max_seq_length_quantile', default=1.0, type=float)
    parser.add_argument('--lineage', default=None, type=str, choices=["pdm09", "Victoria", "Yamagata", "seasonal"])
    parser.add_argument('--split_by', default="month", type=str, choices=["year", "month", "day"])
    parser.add_argument('--identity_keys', nargs="+", default=[], type=str, choices=["lineage", "host", "location", "age", "subtype"])
    parser.add_argument('--min_count', default=0, type=int)
    parser.add_argument('--min_seq_length', default=-1, type=int)
    parser.add_argument('--max_seq_length', default=-1, type=int)
    parser.add_argument('--continent', default=None, type=str, help="Only keep the data from this continent.")
    parser.add_argument('--country', default=None, type=str, help="Only keep the data from country.")
    parser.add_argument('--filter_passage', action='store_true')
    parser.add_argument('--filter_passage_remove_nan', action='store_true')
    parser.add_argument('--filter_passage_original_only', action='store_true')
    parser.add_argument('--separate_region', default=None, type=int, help="The order the locations.")
    parser.add_argument('--continent_to_countries_file', default=None, type=str)

    args = parser.parse_args()

    if args.continent_to_countries_file is not None:
        continent_to_countries = {x[0]: x[1] for x in json.load(open(args.continent_to_countries_file))}
    else:
        continent_to_countries = None

    if args.start_date is None:
        setattr(args, "start_date", START_DATE[args.antigen])

    save_path, save_ids_path = customize_save_path(args)

    # Read meta data
    meta_data = read_meta(args.meta_data_path, meta_data_acc_id_field[args.antigen], delimiter="\t" if args.antigen == "cov" else ",")
    if args.end_date is None:
        print("Automatic detected last date:", (datetime.datetime(*get_max_collection_date(meta_data, meta_data_collection_date_field[args.antigen])) + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
        args.end_date = "9999-12-31"
   
    sequences = read_fasta(args.sequences_path, quiet=True)
    if args.min_seq_length >= 0:
        min_seq_length = args.min_seq_length
    else:
        min_seq_length = np.quantile(np.asarray([len(x[1]) for x in sequences]), args.min_seq_length_quantile)
    if args.max_seq_length >= 0:
        max_seq_length = args.max_seq_length
    else:
        max_seq_length = np.quantile(np.asarray([len(x[1]) for x in sequences]), args.max_seq_length_quantile)
    print("min_seq_length:", min_seq_length, ", max_seq_length:", max_seq_length)

    reference_seq_length, reference_seq_count = Counter([len(x[1]) for x in sequences]).most_common(1)[0]
    print("Reference sequence length %d (%g percent)" % (reference_seq_length, reference_seq_count / len(sequences)))

    if args.antigen == "flu":
        id2seq, id2desc = build_id2seq_flu(sequences)
    elif args.antigen == "cov":
        id2seq, id2desc = build_id2seq_cov(sequences)

    skip_bad_day = 0
    skip_location = 0
    skip_not_meta = 0
    skip_host = 0
    skip_subtype = 0
    skip_lineage = 0
    skip_bad_seqs = 0
    skip_passage = 0
    sequences = []
    temporal_sequences = dict()
    # location_temporal_sequences = dict()
    location_stats = defaultdict(int)

    if args.split_by == "day":
        date_transform = date_to_day
    elif args.split_by == "month":
        date_transform = date_to_month
    elif args.split_by == "year":
        date_transform = date_to_year

    start_date = args.start_date
    end_day = date_transform(args.end_date, start_day=start_date)
    start_day = date_transform(args.start_date, start_day=start_date)
    assert end_day > start_day

    all_subtypes = set()

    for i, pid in enumerate(tqdm(id2seq)):

        seg_id = pid
        if seg_id not in meta_data:
            skip_not_meta += 1
            continue
        
        collection_date = meta_data[seg_id][meta_data_collection_date_field[args.antigen]]
        day = date_transform(collection_date, start_day=start_date)
        year_float = year_fraction(collection_date)
        if day is None or day < start_day or day >= end_day or year_float is None:
            skip_bad_day += 1
            continue

        host = meta_data[seg_id]["Host"].lower()
        if args.host:
            if args.host[:4] != "non-" and host != args.host: # e.g., Human
                skip_host += 1
                continue
            if args.host[:4] == "non-" and host == args.host.split("non-")[1]: # e.g., non-Human
                skip_host += 1
                continue

        if args.antigen == "flu":
            subtype = meta_data[seg_id]["Subtype"].lower()
            lineage = meta_data[seg_id]["Lineage"].lower().strip()
            if args.subtype and not check_flu_subtype(subtype, args.subtype):
                skip_subtype += 1
                continue
            if args.lineage and lineage != args.lineage.lower():
                skip_lineage += 1
                continue
        
        if "location" in args.identity_keys:
            if args.country is not None:
                location = get_location(meta_data[seg_id], 2, continent_to_countries)
            elif args.continent is not None:
                location = get_location(meta_data[seg_id], 1, continent_to_countries)
            else:
                location = get_location(meta_data[seg_id], args.separate_region, continent_to_countries)

            if not location:
                skip_location += 1
                continue
            
        if args.continent is not None:
            if location[0] != args.continent:
                skip_location += 1
                continue
        
        if args.country is not None:
            if len(location) <= 1:
                skip_location += 1
                continue
            country = location[:2]
            if country[1] != args.country:
                skip_location += 1
                continue
        
        if args.filter_passage:
            if not filter_passage(meta_data[seg_id]["Passage_History"], args.filter_passage_remove_nan, args.filter_passage_original_only):
                # print(meta_data[seg_id]["Passage_History"])
                skip_passage += 1
                continue
        
        seq = id2seq[pid]
        if len(seq) < min_seq_length or len(seq) > max_seq_length:
            skip_bad_seqs += 1
            continue
        
        time_bin = int(day) // args.time_interval

        time_float = day / args.time_interval
        if time_bin not in temporal_sequences:
            temporal_sequences[time_bin] = defaultdict(list)
    
        key = (seq, )
        for k in args.identity_keys:
            if k == "host":
                key = key + (host, )
            elif k == "subtype":
                key = key + (subtype.replace(" / ", "_"), )
            elif k == "lineage":
                key = key + (lineage, )
            elif k == "location":
                key = key + (location, )
        temporal_sequences[time_bin][key].append((pid, year_float))

    print("skip_bad_day", skip_bad_day)
    print("skip_host", skip_host)
    print("skip_subtype", skip_subtype)
    print("skip_bad_seqs", skip_bad_seqs)
    print("skip_not_meta", skip_not_meta)
    print("skip_passage", skip_passage)

    skip_small_clades = 0
    skip_small_clades_seqs = 0
    tot_sample_num = 0

    temporal_seq_count = {}
    temporal_size = {}

    last_time_bin = max(temporal_sequences.keys())
    print("last_time_bin", last_time_bin)

    for t in temporal_sequences:
        seq_cnt = {seq: len(temporal_sequences[t][seq]) for seq in temporal_sequences[t]}
        tot_cnt = sum(list(seq_cnt.values()))
        temporal_size[t] = tot_cnt
        if args.remove_min_size > 0 and tot_cnt < args.remove_min_size:
            skip_small_clades_seqs += tot_cnt
            skip_small_clades += 1
            continue
        tot_sample_num += len(seq_cnt)
        temporal_seq_count[t] = {seq: cnt / tot_cnt for seq, cnt in seq_cnt.items() if cnt >= args.min_count}
    
    print("Remove small clade", skip_small_clades, "and seqs", skip_small_clades_seqs)
    print("sample number", tot_sample_num, "average over time", tot_sample_num / len(temporal_seq_count))
    # print("Remove small clade", skip_small_clades, "and seqs", skip_small_clades_seqs)
    with open(save_path, "w") as fout:
        print("Writing to %s" % save_path)
        for t in temporal_seq_count:
            for seq in temporal_seq_count[t]:
                seqstr = seq[0]
                infostr = []
                for ind_key, value in zip(args.identity_keys, seq[1:]):
                    if ind_key == "location":
                        for loc_name, loc in zip(regions_level[:args.separate_region], value):
                            infostr.append("%s=%s" % (loc_name, loc))
                        infostr.append("%s=%s" % (ind_key, "/".join(value)))
                    else:
                        infostr.append("%s=%s" % (ind_key, value))
                # for ind_key, value in zip(args.identity_keys, seq[1:]):
                #     infostr.append("%s=%s" % (ind_key, value))
                infostr = "|".join(infostr)
                if len(infostr) > 0:
                    infostr = "|" + infostr

                acc_id = temporal_sequences[t][seq][0][0] # just a random one
                
                time_float = np.mean([x[1] for x in temporal_sequences[t][seq]])
                time_float_std = np.std([x[1] for x in temporal_sequences[t][seq]])

                fout.write(">%s time_bin=%d|year_frac=%.6f|freq=%g|bin_size=%d%s\n%s\n\n" % (acc_id, t, time_float, temporal_seq_count[t][seq], temporal_size[t], infostr, seqstr))
                # fout.write(">%s time_bin=%d|year_frac=%.6f|freq=%g|count=%d|bin_size=%d%s\n%s\n\n" % (acc_id, t, time_float, temporal_seq_count[t][seq], temporal_seq_count[t][seq]*temporal_size[t], temporal_size[t], infostr, seqstr))

    pickle.dump({t: temporal_sequences[t] for t in temporal_seq_count}, open(save_ids_path, "wb"))
