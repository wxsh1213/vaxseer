
start_year="$1"
end_year="$2"

for year in `seq $start_year $end_year`
do
    for subtype in "a_h3n2" "a_h1n1"
    do
        echo "Prepare data for $year, $subtype"
        month="02"
        meta_data_path="gisaid/metadata.csv"
        sequences_path="gisaid/ha.fasta"
        save_dir="gisaid/ha_processed"

        year_minus_three=`(expr $year - 3)`
        python process_fasta.py --time_interval 9999 \
            --start_date "$year_minus_three"-"$month" \
            --end_date $year-"$month" \
            --subtype $subtype --host human --split_by month \
            --meta_data_path $meta_data_path \
            --sequences_path $sequences_path \
            --save_dir $save_dir \
            --remove_min_size 1000 \
            --min_count 5    
    done
done