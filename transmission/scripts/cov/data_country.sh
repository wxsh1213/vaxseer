cd ../data

continent_to_countries_file="../transmission/data/continent2countries/cov/top32_minCnt1000.json"
meta_data_path="gisaid/raw_cov/GISAID_hcov-19_ids_2023_11_21_05_00/cov_meta_2023_11_21_05_00.tsv"
sequences_path="gisaid/raw_cov/GISAID_hcov-19_ids_2023_11_21_05_00/rbd_from_meta_2023_11_21_05_00.fasta"
save_dir="gisaid/cov/spike_rbd_processed_countries/"

for year_and_month in "2021-07" "2021-10" "2022-01" "2022-04"
do
    echo $year_and_month
    python process_fasta.py \
        --antigen cov \
        --meta_data_path $meta_data_path \
        --sequences_path $sequences_path \
        --save_dir $save_dir  \
        --time_interval 1 \
        --start_date 2020-01 \
        --end_date $year_and_month \
        --remove_min_size 100 \
        --host human \
        --split_by month \
        --identity_keys location \
        --separate_region 2 \
        --continent_to_countries_file $continent_to_countries_file \
        --min_seq_length 223 --max_seq_length 223
done

# For evaluation
python process_fasta.py \
    --antigen cov \
    --meta_data_path $meta_data_path \
    --sequences_path $sequences_path \
    --save_dir $save_dir  \
    --time_interval 3 \
    --start_date 2020-01 \
    --end_date 2023-12 \
    --remove_min_size 100 \
    --host human \
    --split_by month \
    --identity_keys location \
    --separate_region 2 \
    --continent_to_countries_file $continent_to_countries_file \
    --min_seq_length 223 --max_seq_length 223

python split_by_time.py gisaid/cov/spike_rbd_processed_countries/2020-01_to_2023-12_3M/all/human_minBinSize100_minLen223_maxLen223_location_region2.fasta

# # For training oracle models
python process_fasta.py \
    --antigen cov \
    --meta_data_path $meta_data_path \
    --sequences_path $sequences_path \
    --save_dir $save_dir  \
    --time_interval 1 \
    --start_date 2019-12 \
    --end_date 2023-12 \
    --remove_min_size 100 \
    --host human \
    --split_by month \
    --identity_keys location \
    --separate_region 2 \
    --continent_to_countries_file $continent_to_countries_file \
    --min_seq_length 223 --max_seq_length 223

python split_by_location.py gisaid/cov/spike_rbd_processed_countries/2020-01_to_2023-12_3M/all/human_minBinSize100_minLen223_maxLen223_location_region2.fasta
