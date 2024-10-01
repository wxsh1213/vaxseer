#!/bin/bash

month="02"

for year in `seq 2012 2021`
do
    for subtype in "a_h3n2" "a_h1n1"
    do
        echo ">>>" $year $subtype

        index=`expr \( $year - 2018 \) \* 2 + 30`
        three_years_before=`(expr $year - 3)`

        working_directory="../runs/pipeline/"$year"-"$month"/$subtype/vaccine_set=who___virus_set=$three_years_before-"$month"-$year-"$month""

        ground_truth_hi_path="../data/antigenicity/hi_processed/before_2023-04/"$subtype"_hi_folds.csv"
        ground_truth_prob_path="../data/gisaid/ha_processed/2003-10_to_2023-04_6M/$subtype/human_minBinSize100_lenQuantile0.2_bins/$index.fasta"

        sequence_file="../data/gisaid/raw/ha.fasta"

        nohup python pipeline/sweep_pairs.py --dominance_prediction_dir $working_directory"/dominance_prediction" --hi_prediction_dir $working_directory"/vaccine_virus_pairs/prediction" --save_dir $working_directory"/vaccine_scores" --ground_truth_hi_path $ground_truth_hi_path --ground_truth_prob_path  $ground_truth_prob_path --sequence_file $sequence_file > nohup.sweep_pairs.$subtype.$year.log 2>&1 &

    done
done
