
This is the codebase for paper: [Predicting sub-population specific viral evolution](https://openreview.net/forum?id=Mae23iEqPS)

# Data

## 1. download HA sequences and metadata from GISAID

First, you might need to register for the access of [GISAID](https://gisaid.org/). Download the HA protein sequences and metadata from EpiFlu/EpiCov. The accession ids of proteins used in our experiments could be found in `data/gisaid/ha_acc_ids.csv` for FLU, and `data/gisaid/cov_acc_ids.csv` for COV.

For COV, the sequence of Spike RBD is extracted from the metadata:

```
metadata_path="../data/gisaid/raw_cov/GISAID_hcov-19_ids_2023_11_21_05_00/cov_meta_2023_11_21_05_00.tsv"
python scripts/cov/from_metadata_to_fasta.py $metadata_path
```

## 2. build training/evaluation data

After downloading the raw sequence and metadata, the following scripts are provided for processing the training data. 

Assume that the protein sequences and metadata for FLU are saved in `data/gisaid/raw/ha.fasta` and `data/gisaid/raw/metadata.csv`. The protein sequences and metadata for COV are saved in `data/gisaid/raw_cov/GISAID_hcov-19_ids_2023_11_21_05_00/rbd_from_meta_2023_11_21_05_00.fasta` and `data/gisaid/raw_cov/GISAID_hcov-19_ids_2023_11_21_05_00/cov_meta_2023_11_21_05_00.tsv`.

```
# FLU, continent:
bash scripts/flu/data_continents.sh

# FLU, country:
bash scripts/flu/data_country.sh

# COV, continent
bash scripts/cov/data_continents.sh

# COV, country:
bash scripts/cov/data_country.sh

```

# Training

The following scripts are provided for training transmission models at the continent level and hierarchical transmission models at the country level.

```
## FLU, continent
year="2018"
devices="2,3"
bash scripts/flu/train_transmission.sh $year $devices

## FLU, country
year="2018"
devices="6,"
# G2G
bash scripts/flu/train_hierachy.sh $year $devices "continent_to_continent"
# G2L
bash scripts/flu/train_hierachy.sh $year $devices "country_to_continent"

## COV, continent
year="2021"
month="07"
devices="0,1,2,3"
bash scripts/cov/train_transmission.sh $year $month $devices

## COV, country
year="2021"
month="07"
devices="4,6"
# G2G
bash scripts/cov/train_hierarchy.sh $year $month $devices "continent_to_continent"
# G2L
bash scripts/cov/train_hierarchy.sh $year $month $devices "country_to_continent"
```

# Evaluation

## Training orcle models

To evaluate the quality of the generated sequences, we train Oracle models for each sub-population:

```
# FLU, continents
devices="6,"
location="asia"
random_seed="1005" # 1005 / 1213 / 529
bash scripts/flu/train_oracle_continents.sh $devices $location $random_seed

# FLU, countries
devices="6,"
random_seed="1005"
bash scripts/flu/train_oracle_countries.sh $devices $random_seed

# COV, continents
devices="6,"
location="asia"
bash scripts/cov/train_oracle_continents.sh $location $devices

# COV, countries
devices="6,"
random_seed="1213"
bash scripts/cov/train_oracle_countries.sh $devices $random_seed

```


## Evaluate NLL

### Flu continent-level
```
year="2018"
devices="7,"

for temperature in "0.2" "0.4" "0.6" "0.8" "1.0"
do
    bash scripts/flu/eval_transmission.sh $year $devices $temperature
done

```
### Flu country-level

```
year="2018"
devices="6,"

for temperature in "0.2" "0.4" "0.6" "0.8" "1.0"
do
    bash scripts/flu/eval_hierachy.sh $year $devices $temperature continent_to_continent
    
    bash scripts/flu/eval_hierachy.sh $year $devices $temperature country_to_continent
done
```

### COV continent-level

```
year="2021"
month="07"
devices="7,"

bash scripts/cov/eval_transmission.sh $year $month $devices

```

### COV country-level
```
year="2021"
month="07"
devices="6,"

bash scripts/cov/eval_hierachy.sh $year $month $devices continent_to_continent
bash scripts/cov/eval_hierachy.sh $year $month $devices country_to_continent

```

## Generate sequences

```
## FLU, continent:
year="2018"
devices="4,"
bash scripts/flu/generate_transmission.sh $year $devices "0.2 0.4 0.6 0.8 1.0"


## FLU, country:
year="2018"
devices="6,"
bash scripts/flu/generate_hierarchy.sh $year $devices "0.2 0.4 0.6 0.8 1.0" continent_to_continent
bash scripts/flu/generate_hierarchy.sh $year $devices "0.2 0.4 0.6 0.8 1.0" country_to_continent


## COV, continent:
year="2021"
month="07"
devices="1,"
bash scripts/cov/generate_transmission.sh $year $month $devices "0.2 0.4 0.6 0.8 1.0"
# Beam search
bash scripts/cov/generate_transmission_beam_search.sh $year $month $devices

## COV, country:
year_and_month="2021-07"
devices="5,"
# G2G
bash scripts/cov/generate_hierarchy.sh $year_and_month $devices "0.2 0.4 0.6 0.8 1.0" continent_to_continent
# G2L
bash scripts/cov/generate_hierarchy.sh $year_and_month $devices "0.2 0.4 0.6 0.8 1.0" country_to_continent
# Beam search
bash scripts/cov/generate_hierarchy_beam_search.sh $year_and_month $devices continent_to_continent
bash scripts/cov/generate_hierarchy_beam_search.sh $year_and_month $devices country_to_continent

```

## Evaluate reverse NLL

### FLU, continent

```
year=2018
devices=7,
seed=1005 # Random seed for the oracle model

for temperature in "0.2" "0.4" "0.6" "0.8" "1.0"
do
    bash scripts/flu/rev_nll_continents.sh $year $devices $temperature $seed
done
```

### FLU, country
```
year="2018"
devices=7,

for temperature in "0.2" "0.4" "0.6" "0.8" "1.0"
do
    for oracle_model in 1 2 3
    do
        bash scripts/flu/rev_nll_countries.sh $year $devices $temperature $oracle_model
    done
done
```
### COV, continent
```
year="2021"
month="07"
devices=7,

oracle=1 # 1 2 3

for temperature in "0.2" "0.4" "0.6" "0.8" "1.0"
do
    for oracle_model in 1 2 3
    do 
        bash scripts/cov/rev_nll_continents.sh $year $month $devices $temperature $oracle_model
    done
done
```
### COV, country
```
year=2021
month=07
devices="6,"

for temperature in "0.2" "0.4" "0.6" "0.8" "1.0"
do
    for oracle_model in 1 2 3
    do
        bash scripts/cov/rev_nll_countries.sh $year $month $temperature $devices $oracle_model continent_to_continent
        # or
        # bash scripts/cov/rev_nll_countries.sh $year $month $temperature $devices $oracle_model country_to_continent
    done
done
```

## Coverage for COV

See `scripts/cov/coverage.ipynb`.

# Ablation study

## Train models with varied groups

Set `--continent_to_country_mapping_file` to different options in `transmission/data/continent2countries/flu` or `transmission/data/continent2countries/cov` will modify the group configurations.

`--topk_eigen $k`: Specifies the number of largest eigenvalues and eigenvectors to use in the transmission matrix, where $k is the selected value.

## Train models with 

# Results visualization

See `scripts/flu/visualize_flu_*.ipynb` for FLU and `scripts/cov/visualize_cov_*.ipynb` for COV.