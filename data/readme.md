# Download HA sequences and metadata from GISAID

First, you might need to register for the access of [GISAID](https://gisaid.org/). Download the HA protein sequences and metadata from EpiFlu. The accession ids of proteins used in our experiments could be found in `gisaid/ha_acc_ids.csv`.

# Build training data for dominance predictors

Following code is used to build the training data collected before `year`-`month` for dominance predictors.

```
year="2018"
month="02"
subtype="a_h3n2" # or a_h1n1
meta_data_path="gisaid/raw/metadata.csv"
sequences_path="gisaid/raw/ha.fasta"
save_dir="gisaid/ha_processed"

python process_fasta.py --time_interval 2 \
    --start_date 2003-10 \
    --end_date "$year"-"$month" \
    --remove_min_size 100 \
    --subtype $subtype --host human --split_by month \
    --meta_data_path $meta_data_path \
    --sequences_path $sequences_path \
    --save_dir $save_dir --min_seq_len 553
    
```

or run `gisaid/batch_prepare_dominance_training.sh 2012 2021` to process all data from 2012 to 2021.

# Building training data for antigenicity predictors

The HI test results are collected from the reports from the [Worldwide Influenza Centre lab](https://www.crick.ac.uk/research/platforms-and-facilities/worldwide-influenza-centre/annual-and-interim-reports). 

We download the reports before February 2023. Following code is used to download reports:

```
python antigenicity/dowload_pdfs.py antigenicity/reports_pdf
```

We convert pdf to xlsx using [tools](https://premium.pdftoexcel.com/). 

Assuming that xlsx files are save under folder `antigenicity/reports_xlsx`, you could run `antigenicity/antigenicity_prepare_data.sh` to extract antigenic analyses tables and vaccine-virus pairs from .xlsx files.

`antigenicity/antigenicity_pipeline.sh` shows the example of building training data for one winter season:
```
year="2018"
month="02"

bash antigenicity_pipeline.sh $year $month
```

You could run `antigenicity/antigenicity_pipeline_batch.sh` to process data from 2012 to 2021.
