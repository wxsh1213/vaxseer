import os, csv, sys
import pandas as pd
from collections import defaultdict
import numpy as np

root_dirs = sys.argv[1:-1]
saving_path = sys.argv[-1]

def normalize_name(name):
    try:
        name = name.replace(" ", "_").replace("-", "_")
        if name.endswith("*"):
            name = name[:-1]

        name_split = name.split("/")
        if len(name_split) >= 4:
            name_split[1] = name_split[1].replace("2", "")
            name_split[1] = name_split[1].replace("3", "")
            name_split[1] = name_split[1].replace("4", "")

            if len(name_split[3]) == 2: # e.g., 02 -> 2002
                if int(name_split[3][0]) <= 2:
                    year = "20" + name_split[3]
                else:
                    year = "19" + name_split[3]
                name_split[3] = year
        
        name = "/".join(name_split)
        if name == "a/california/7/2009": # a/california/07/2009 is the name used in historical vaccine
            name = "a/california/07/2009"
        return name
    except Exception as e:
        print(e)
        print("Fail to normalize name", name)
        return None

pairs = dict()
ref_seqs = set()
virus_names = set()

for root_dir in root_dirs:
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(root, file)
            with open(path, newline='') as csvfile:
                spamreader = csv.reader(csvfile)
                for i, row in enumerate(spamreader):
                    if i == 0:
                        headline = row[1:]
                    else:
                        # print(', '.join(row))
                        virus = row[0]
                        virus_name = normalize_name(virus) # virus.replace(" ", "_").replace("-", "_")
                        if virus_name is None:
                            continue
                        virus_names.add(virus_name)
                        if virus_name not in pairs:
                            pairs[virus_name] = defaultdict(list)
                        # pairs[virus]
                        for ii, (ref_seq, value) in enumerate(zip(headline, row[1:])):
                            if ref_seq: # vaccine name
                                if len(ref_seq.split()[0].split("/")) == 4:
                                    ref_seq = ref_seq.split()[0]
                                elif len(ref_seq.split()[0].split("/")) > 4:
                                    ref_seq = "/".join(ref_seq.split("/")[:4])
                                ref_seq = ref_seq.replace(" ", "_").replace("-", "_")
                                pairs[virus_name][ref_seq].append(value)
                                ref_seqs.add(ref_seq)


loc_alias = {
    "hk": "hong kong",
    "switz": "switzerland",
    "swit": "switzerland",
    "wis": "wisconsin",
    "bris": "brisbane",
    "fin": "finland",
    "nor": "norway",
    "bang": "bangladesh",
    "slo": "slovenia",
    "slov": "slovenia",
    "eng": "england",
    "ny": "new_york",
    "fuj": "fujian_yanping",
    # "fuj": "fujian",
    "mich": "michigan",
    "ann": "annecy",
    "neth": "netherlands",
    "well": "wellington",
    "flor": "florida",
    "serr": "serres",
    "c'church": "christchurch",
    "alab": "alabama",
    "bret": "bretagne",
    "nd": "north_dakota",
    "uru": "uruguay",
    "chch": "christchurch",
    "sing": "singapore",
    "sth_africa": "south_africa",
    "jbg": "johannesburg",
    "bw": "baden_wurttemburg",
    "vic": "victoria",
    "vict": "victoria",
    "stock": "stockholm",
    "sant": "santiago",
    "cal": "california",
    "ca": "california",
    "calif": "california",
    "nh": "new_hampshire",
    "sth afr": "south_africa",
    "hiro": "hiroshima",
    "wy": "wyoming",
    "nth_carol": "north_carolina",
    "camb": "cambodia",
    "jhb": "johannesburg",
    "tri": "trieste",
    "prag": "prague",
    "phil": "philippines",
    "mary": "maryland",
    "bay": "bayern",
    "send": "sendai",
    "s_africa": "south_africa",
    "saust": "south australia",
    "sth_aus": "south_australia",
    "s.australia": "south_australia",
    "pan": "panama",
    "wyom": "wyoming",
    "sth_afr": "south_africa",
    "ala": "alabama",
    "wis___": "wisconsin",
    "glas": "glasgow",
    "johnnesburg": "johannesburg",
    "si": "solomon_islands",
    "beij": "beijing",
    "st._p": "st._petersburg",
    "st._p'burg": "st._petersburg",
    "st_p": "st._petersburg",
    "s._peter": "st._petersburg",
    "st._pet": "st._petersburg",
    "sp": "st._petersburg",
    "n_carolina": "north_carolina",
    "nor_carol": "north_carolina",
    "nc": "north_carolina",
    "fuk": "fukushima",
    "nj": "new_jersey",
    "n_jers": "new_jersey",
    "eg": "egypt",
    "c'ch":"christchurch",
    "scot": "scotland",
    "auck":"auckland",
    "sey":"seychelles",
    "toamas": "toamasina",
    "thess": "thessaloniki",
    "braz": "brazil",
    "mos": "moscow",
    "mad": "madagascar",
    "ire": "ireland",
    "hung": "hungary",
    "s._africa": "south_africa",
    "astr": "astrakhan",
    "astrak": "astrakhan",
    "voro": "voronezh",
    "dr": "dominican_republic",
    "nc": "new_caledonia",
    "gan_baiy": "gansu_baiyin",
    "g_baiyin": "gansu_baiyin",
    "nord_west": "nordrhein_westfalen",
    "civ": "cote_d'ivoire",
    "mal": "malaysia",
    "tehr": "tehran",
    "wash'ton": "washington",
    "for": "formosa",
    "hen_xig": "henan_xigong",
    "cam": "cambodia",
    "shan": "shandong",
    "nieder": "niedersachsen",
    "mass": "massachusetts",
    "barc": "barcelona",
    "fl": "florida",
    "sen": "sendai_h",
    "maur": "mauritius",
    "vall": "valladolid",
    "h'burg": "hamburg",
    "ban": "bangladesh",
    "sich": "sichuan",
    "novo": "novosibirsk",
    "shai": "shanghai",
    "jiang": "jiangsu",
    "alg": "algeria",
    "guad": "guadeloupe",
    "zag": "zagreb",
    "rhode_is": "rhode_island",
    "s_aus": "south_australia",
    "niig": "niigata",
    "mont": "montana",
    "g_m": "guangdong_maonan"

}

vaccine_map = {
    "a/brisbane/299/2011": "ivr_164(a/brisbane/299/2011)",
    "x_261/hong_kong/7127/2014": "a/hong_kong/7127/2014",
    # "a/norway/4465/2017": "a/norway/4465/2016",
    "nymc_x_327/a/kansas/14/17": "nymc_x_327_(a/kansas/14/17)",
    "nymc_x_327/a/kan/14/17": "nymc_x_327_(a/kansas/14/17)",
    "nymc_x_327/a/kans/14/17": "nymc_x_327_(a/kansas/14/17)",
    "nymc_x_327/a/kansas/2014": "nymc_x_327_(a/kansas/14/17)",
    "nymc/x_263b": "nymc_x_263b_(a/hong_kong/4801/2014)",
    "nib_85": "nib_85_(a/almaty/2958/2013)",
    "ivr_197": "ivr_197__(a/south_australia/34/2019)",
    "nymc/x_261": "nymc_x_261_(a/hk/7127/2014)",
    "nib_103": "nib_103_(a/norway/3806/2016)",
    "ivr_197/a/sth_aus/34/19": "ivr_197__(a/south_australia/34/2019)",
    "x_263b/hong_kong/4801/2014": "nymc_x_263b_(a/hong_kong/4801/2014)",
    "x_199/(ri/1/10)": "a/rhode_island/1/2010_(x_199)",
    "b/maur/i_762": "b/mauritius/i_762/2018"
}

special_shorts = { # fix bugs by hand :(
    # "ivr_215": "ivr_215_(a/victoria/2570/2019)",
    "ivr_215": "a/victoria/2570/2019",
    # "x_243": "x_243_(a/south_africa/3626/2013)",
    "x_243": "a/south_africa/3626/2013",  
    "a/cal/7/09": "a/california/07/2009",
    "a/singapore/0019/16": "a/singapore/infimh_16_0019/2016",
    "a/sing/0019/16": "a/singapore/infimh_16_0019/2016",
    "a/sing/19/16": "a/singapore/infimh_16_0019/2016",
    "0019/16/egg_10_6": "a/singapore/infimh_16_0019/2016",
    "0019/16/egg_10_4": "a/singapore/infimh_16_0019/2016",
    "a/c'church/515/18": "a/christchurch/515/2019",
    "a/nantes/1441.0": "a/nantes/1441/2017",
    "a/shan/1219/04": "a/shantou/1219/2004",
    "a/catal/nsvh_2067/22": 'a/catalonia/nsvh161512067/2022',
    "a/slov/134/06": "a/slovakia/134/2006"
}

def rename_ref_strains(ref_names, virus_names):
    virus_names_split = [x.split("/") for x in virus_names]
    
    def vague_search(x):
        matched = []
        x_split = x.split("/")
        for v in virus_names_split:
            if len(x_split) >= 4 and len(v) >= 4 and x_split[0] == v[0] and x_split[2] == v[2] and x_split[3] == v[3]:
            # if len(set(x_split) & set(v)) >= 3:
                # print()/
                matched.append("/".join(list(v)))
            # elif len(x_split) >= 4 and len(v) >= 4 and x_split[1] == v[1] and x_split[2] == v[2] and x_split[3] == v[3]:
                # matched.append("/".join(list(v)))
        return matched
    
    # def vague_match():

    short_name = set()
    name_map = {}
    for ref_name in ref_names:
        ori_ref_name = ref_name

        if ref_name[0] == "(" and ref_name[-1] == ")":
            ref_name = ref_name[1:-1]

        # if len(ref_name.split("/")) == 3:
        #     if ref_name.split("/")[0] == "a":
        #         ref_name = "a/" + ref_name

        special_shorts_found = False
        for ss in special_shorts:
            if ss in ref_name:
                ref_name = special_shorts[ss]
                special_shorts_found = True
        
        if not special_shorts_found:
        # if ref_name == ori_ref_name:
            if len(ref_name.split("/")) == 4:
                if len(ref_name.split("/")[-1]) == 2:
                    year = ref_name.split("/")[-1]
                    if int(year[0]) <= 2:
                        year = "20" + year
                    else:
                        year = "19" + year
                    ref_name = "/".join(ref_name.split("/")[:-1] + [year])
                if ref_name.split("/")[0] != "a" and "a" in ref_name.split("/")[0]: # *a, 2a
                    ref_name = "/".join(["a"] + ref_name.split("/")[1:])
                
                loc = ref_name.split("/")[1].replace("2", "")
                loc = loc.replace("1,4", "")
                loc = loc.replace("1,3", "")
                loc = loc.replace("1", "")
                loc = loc.replace("3", "")
                loc = loc.replace("4", "")
                loc = loc.replace("*", "")
                
                if loc in loc_alias:
                    ref_name = "/".join(ref_name.split("/")[:1] + [loc_alias[loc].replace(" ", "_").replace("-", "_")] + ref_name.split("/")[2:])
                else:
                    ref_name = "/".join(ref_name.split("/")[:1] + [loc] + ref_name.split("/")[2:])

        ref_name = ref_name.strip("/").strip()
        if ref_name in virus_names:
            name_map[ori_ref_name] = ref_name
        elif ref_name in vaccine_map:
            name_map[ori_ref_name] = vaccine_map[ref_name]
        else:
            pass
    return name_map


name_map = rename_ref_strains(ref_seqs, virus_names)

for x in ref_seqs:
    if x not in name_map:
        print("Incorrect vaccine name:", x)

ref_seqs = set([name_map[x] for x in ref_seqs if x in name_map])

new_pairs = {}
for v in pairs:
    new_pairs[v] = defaultdict(list)
    # print(pairs[v])
    for ref in pairs[v]:
        if ref not in name_map:
            continue
        new_pairs[v][name_map[ref]] += pairs[v][ref]
pairs = new_pairs
    # {ref_seqs[k]: pairs[virus][k] for k in pairs[virus]}
    # for refseq in pairs[virus]:


# exit()

num_pairs = 0
number_of_vaccine = []
for virus in pairs:
    number_of_vaccine.append(len(pairs[virus]))
    num_pairs += len(pairs[virus])

# print("How many pairs do we have?", num_pairs)
# print(sum(number_of_vaccine) / len(number_of_vaccine))

vaccine_set_2 = set()
with open(saving_path, "w") as fout:
    spamwriter = csv.writer(fout)
    spamwriter.writerow(["virus", "reference", "hi"])

    for ids in pairs:
        for vaccine in pairs[ids]:
    # for ids in subset:
    #     for vaccine in pairs[ids]:
    #         if vaccine not in id2record: 
    #             continue
            values = []
            for value in pairs[ids][vaccine]:
                if value == "nd" or value == "nt":
                    continue
                elif value == "<" or value == "<40":
                    values.append(40.0)
                elif value == ">5120":
                    values.append(5120.0) # TODO?
                else:
                    try:
                        value = float(value)
                        if value <= 0:
                            continue
                        values.append(float(value))
                        
                    except Exception as e:
                        print("Unrecognize value: %s" % value)
            
            if len(values) > 0:
                values = np.asarray(values)
                ave_values = np.exp(np.log(values).mean()) # .exp()
                spamwriter.writerow([ids, vaccine, ave_values])
                vaccine_set_2.add(vaccine)
