#!/bin/bash

month="02"

for year in `seq 2012 2014`
do
    for subtype in "a_h1n1" "a_h3n2"
    do
        echo ">>" $year $subtype

        index=`expr \( $year - 2018 \) \* 2 + 30`
        history_index=`expr $index - 2`
        year_plus_one=`expr \( $year + 1 \) `

        working_directory="../runs/pipeline/"$year"-$month/$subtype/vaccine_set=`(expr $year - 3)`-$month-$year-"$month"___virus_set=`(expr $year - 3)`-"$month"-$year-"$month""

        virus_fasta_path="../data/gisaid/ha_processed/`(expr $year - 3)`-"$month"_to_"$year"-"$month"_9999M/$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.fasta"

        # History
        dominance_history_results="../data/gisaid/ha_processed/2003-10_to_2023-04_6M/$subtype/human_minBinSize100_lenQuantile0.2_bins/$history_index.fasta"
        save_dir="$working_directory/dominance_prediction/history_$history_index"

        mkdir -p $save_dir

        python baselines/dominance_from_fasta.py --source_fasta_path $dominance_history_results --target_fasta_path $virus_fasta_path --save_path "$save_dir/test_results.csv"

    done
done

