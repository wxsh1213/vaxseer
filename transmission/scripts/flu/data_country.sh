cd ../data

meta_data_path="gisaid/raw/metadata.csv"
sequences_path="gisaid/raw/ha.fasta"
save_dir="gisaid/flu/ha_processed_countries"
continent_to_countries_file="../transmission/data/continent2countries/flu/top16_minCnt1000.json"

for year in "2016" "2017" "2018"
do
    echo $year

    python process_fasta.py \
        --antigen flu \
        --time_interval 2 \
        --start_date 2003-10 \
        --end_date "$year"-02 \
        --subtype a_h3n2 \
        --remove_min_size 10 \
        --host human \
        --split_by month \
        --identity_keys location \
        --separate_region 2 \
        --continent_to_countries_file $continent_to_countries_file \
        --meta_data_path $meta_data_path \
        --sequences_path $sequences_path \
        --save_dir $save_dir --min_seq_length 553 --max_seq_length 566
done

# For training the oracle models:
python process_fasta.py \
    --antigen flu \
    --time_interval 6 \
    --start_date 2003-10 \
    --end_date 2023-04 \
    --subtype a_h3n2 \
    --remove_min_size 10 \
    --host human \
    --split_by month \
    --identity_keys location \
    --separate_region 2 \
    --continent_to_countries_file $continent_to_countries_file \
    --meta_data_path $meta_data_path \
    --sequences_path $sequences_path \
    --save_dir $save_dir --min_seq_length 553 --max_seq_length 566

python split_by_time.py gisaid/flu/ha_processed_countries/2003-10_to_2023-04_6M/a_h3n2/human_minBinSize10_minLen553_maxLen566_location_region2.fasta
python split_by_location.py gisaid/flu/ha_processed_countries/2003-10_to_2023-04_6M/a_h3n2/human_minBinSize10_minLen553_maxLen566_location_region2.fasta