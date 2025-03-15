cd ../data

meta_data_path="gisaid/raw/metadata.csv"
sequences_path="gisaid/raw/ha.fasta"
save_dir="gisaid/flu/ha_processed_continents"

# For training:
for year in "2015" "2016" "2017" "2018"
do
    for continent in "north_america" "asia" "europe" "oceania" "south_america" "africa"
    do
        echo $year $continent

        python process_fasta.py \
            --time_interval 2 \
            --start_date 2003-10 \
            --end_date "$year"-02 \
            --remove_min_size 10 \
            --subtype a_h3n2 \
            --host human \
            --split_by month \
            --continent $continent \
            --identity_keys location \
            --meta_data_path $meta_data_path \
            --sequences_path $sequences_path \
            --save_dir $save_dir --min_seq_length 553
    done
done


# For training oracle models:
for continent in "north_america" "asia" "europe" "oceania" "south_america" "africa"
do
    echo $continent

    python process_fasta.py \
        --time_interval 2 \
        --start_date 2003-10 \
        --end_date 2023-04 \
        --remove_min_size 10 \
        --subtype a_h3n2 \
        --host human \
        --split_by month \
        --continent $continent \
        --identity_keys location \
        --meta_data_path $meta_data_path \
        --sequences_path $sequences_path \
        --save_dir $save_dir --min_seq_length 553
done


# For evaluations:
for continent in "north_america" "asia" "europe" "oceania" "south_america" "africa"
do
    echo $continent
    python process_fasta.py \
        --time_interval 6 \
        --start_date 2003-10 \
        --end_date 2023-04 \
        --remove_min_size 10 \
        --subtype a_h3n2 \
        --host human \
        --split_by month \
        --continent $continent \
        --identity_keys location \
        --meta_data_path $meta_data_path \
        --sequences_path $sequences_path \
        --save_dir $save_dir --min_seq_length 553 
        
    python split_by_time.py $save_dir/2003-10_to_2023-04_6M/a_h3n2/continents/human_minBinSize10_minLen553_"$continent"_location.fasta
done