import numpy as np
from collections import Counter, defaultdict
import pandas as pd
import json

raw_meta_data = pd.read_csv("../../data/gisaid/raw/metadata_20250930.csv")
raw_meta_data = raw_meta_data[raw_meta_data["Host"] == "Human"]
raw_meta_data = raw_meta_data.drop_duplicates()
raw_meta_data = raw_meta_data[["Isolate_Id", "Collection_Date", "Subtype", "Location"]]
raw_meta_data = raw_meta_data.drop_duplicates()
raw_meta_data["Location"] = raw_meta_data["Location"].fillna("")
raw_meta_data["country"] = [x.split("/")[1].strip() if len(x.split("/")) >= 2 else x for x in raw_meta_data["Location"] ]
raw_meta_data_us = raw_meta_data[raw_meta_data["country"] == "United States"]

year2subtype_counting = defaultdict(dict)

for year in range(2012, 2025):
    print(year)
    start_date = np.datetime64(f"{year}-10")
    end_date = np.datetime64(f"{year+1}-04")
    
    collection_date = []
    for date in raw_meta_data_us["Collection_Date"]:
        if len(date.split("-")) < 2:
            collection_date.append(False)
        else:
            _year, _month = date.split("-")[:2]
            if np.datetime64(_year+"-"+_month) >= start_date and np.datetime64(_year+"-"+_month) < end_date:
                collection_date.append(True)
            else:
                collection_date.append(False)
    subtype_count = dict(Counter(raw_meta_data_us[np.asarray(collection_date)]["Subtype"]))
    print("A/H3N2", subtype_count["A / H3N2"])
    print("A/H1N1", subtype_count["A / H1N1"])
    year2subtype_counting[year]["h3n2"] = subtype_count["A / H3N2"]
    year2subtype_counting[year]["h1n1"] = subtype_count["A / H1N1"]

print(year2subtype_counting)