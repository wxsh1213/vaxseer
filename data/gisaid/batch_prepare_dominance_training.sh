start_year="$1"
end_year="$2"

for year in `seq $start_year $end_year`
do
    for subtype in "a_h3n2" "a_h1n1"
    do
        echo "Prepare data for $year, $subtype"
        month="02"

        meta_data_path="gisaid/raw/metadata_20240630.csv"
        sequences_path="gisaid/raw/ha_submit_before_2024-06-30.fasta"
        save_dir="gisaid/ha_processed_20240630"

        python process_fasta.py --time_interval 2 \
            --start_date 2003-10 \
            --end_date "$year"-"$month" \
            --remove_min_size 100 \
            --subtype $subtype --host human --split_by month \
            --meta_data_path $meta_data_path \
            --sequences_path $sequences_path \
            --save_dir $save_dir --min_seq_len 553
    done
done