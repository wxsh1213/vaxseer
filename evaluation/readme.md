
# Prepare retrospective evaluation data

## Building candidate vaccine and circulating virus HA sequences.

In our paper, we retrospectively evaluate the performance of our models from 2012 to 2021. 

First, we need to build the candidate vaccine set and circulating viruses set, which consists of HA protein sequences with occurance larger than 5 in the past three years.
For example, if you want to build the candidate vaccine set for A/H3N2 and 2018 winter season,   


```
cd ../data

subtype="a_h3n2" # or a_h1n1
meta_data_path="gisaid/metadata.csv"
sequences_path="gisaid/ha.fasta"
save_dir="gisaid/ha_processed"

python process_fasta.py --time_interval 9999 \
    --start_date 2015-02 \
    --end_date 2018-02 \
    --subtype $subtype --host human --split_by month \
    --meta_data_path $meta_data_path \
    --sequences_path $sequences_path \
    --save_dir $save_dir \
    --remove_min_size 1000 \
    --min_count 5
```

or run `data/batch_prepare_eval_vaccines_and_viruses.sh` to process from 2012 to 2021.

To evaluate the performance based on ground-truth coverage scores, run following code to get dominance of viral strains in all seasons.

```
subtype="a_h3n2" # or a_h1n1
meta_data_path="gisaid/metadata.csv"
sequences_path="gisaid/ha.fasta"
save_dir="gisaid/ha_processed"

python process_fasta.py --time_interval 6 \
    --start_date 2003-10 \
    --end_date 2023-04 \
    --subtype $subtype --host human --split_by month \
    --meta_data_path $meta_data_path \
    --sequences_path $sequences_path \
    --remove_min_size 100 \
    --save_dir $save_dir

python split_by_time.py $save_dir/2003-10_to_2023-04_6M/$subtype/human_minBinSize100_lenQuantile0.2.fasta
```

## Building HA sequences for historical vaccine strains.

To calculte the coverage scores for historical recommended vaccines, we need to find the HA sequences for the recommended vaccines. The strains recommended by the WHO could be found in `data/recommended_vaccines_from_gisaid.csv`.

```
python get_ha_seqs_for_vaccine_strains.py --sequences_path $sequences_path
```

# Predict the coverage scores by VaxSeer

`pipeline/run_vaxseer.sh` provides an example of running the domiance predictor and antigenicity predictor by VaxSeer.

```
subtype="a_h3n2"
year="2018"
device="0"

bash pipeline/run_vaxseer.sh $subtype $year $device

```

To run other baselines in dominance prediction and antigenicity predction, could refer to scripts in `pipeline/baselines`.


To calcualte the coverage scores from predicted dominance and antigenicty, you could run `evaluation/pipeline/run_sweep_comb.sh`, which will run over all combination of dominance and antigenicty predictor, and calculate the ground-truth coverage scores.

```
# calculate the coverage scores for candidate vaccines.
bash pipeline/run_sweep_comb.sh

# calculate the coverage scores for historical recommended vaccines.
bash pipeline/run_sweep_comb_who.sh
```

# Visualization for results

The visualization codes for all figures in our paper are included in `scripts/evaluation/jupyter_notebooks_figs`.

