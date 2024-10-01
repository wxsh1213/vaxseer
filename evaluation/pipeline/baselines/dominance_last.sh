#!/bin/bash

month="02"

for year in `seq 2012 2021`
do
    for subtype in "a_h1n1" "a_h3n2"
    do
        echo ">>" $year $subtype

        index=`expr \( $year - 2018 \) \* 2 + 30`
        history_index=`expr $index - 2`
        year_plus_one=`expr \( $year + 1 \) `

        # 6M
        year_minus_one=`expr \( $year - 1 \) `
        virus_fasta_path="../data/gisaid/ha_processed/"$year_minus_one"-08_to_"$year"-02_9999M/$subtype/human_minBinSize1_lenQuantile0.2.fasta"
        working_directory="../runs/pipeline/"$year"-$month/$subtype/vaccine_set=`(expr $year - 3)`-$month-$year-"$month"___virus_set=last_6M"

        # History distributions
        history_month="6"
        max_history_index=$(( ($year - 2003) * 6 + ($month - 10) / 2 - 1 ))
        min_history_index=$(( $max_history_index - $history_month / 2 + 1 ))

        dominance_history_results=""
        for history_index in `seq $min_history_index $max_history_index`
        do
            dominance_history_results="$dominance_history_results /data/rsg/nlp/wenxian/vaxseer/data/gisaid/ha_processed/2003-10_to_2023-04_2M/$subtype/human_minBinSize10_lenQuantile0.2_bins/$history_index.fasta"
        done
        echo $dominance_history_results
        
        save_dir="$working_directory/dominance_prediction/history_"$history_month"M"

        mkdir -p $save_dir

        python baselines/dominance_from_fasta.py --source_fasta_path $dominance_history_results --target_fasta_path $virus_fasta_path --save_path "$save_dir/test_results.csv"

    done
done

