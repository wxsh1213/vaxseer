
ckpt_saving_root_dir="../runs/flu_lm_aa_subs"

month="02"
time_inverval="2M"

for year in `seq 2015 2021`
do
    for subtype in "a_h1n1" "a_h3n2"
    do
        echo ">>" $year $subtype

        year_minus_three=`(expr $year - 3)`
        index=`expr \( $year - 2018 \) \* 2 + 30`

        testing_time=$index

        if [[ $time_inverval == "2M" ]];
        then
            min_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 5`
            max_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 7`
        else
        # For 6M:
            min_testing_time=$testing_time
            max_testing_time=$testing_time
        fi

        # Testing viruses = 3Y
        working_dir="../runs/pipeline/$year-"$month"/$subtype/vaccine_set=$year_minus_three-"$month"-$year-"$month"___virus_set=$year_minus_three-"$month"-$year-"$month""

        train_fasta_path="../data/gisaid/ha_processed/2003-10_to_$year-"$month"_"$time_inverval"/$subtype/human_minBinSize100_lenQuantile0.2.fasta"
        testing_viruses_path="../data/gisaid/ha_processed/$year_minus_three-"$month"_to_$year-"$month"_9999M/$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.fasta"
        ckpt_saving_dir="$ckpt_saving_root_dir/2003-10_to_$year-"$month"_"$time_inverval"/$subtype/human_minBinSize100_lenQuantile0.2"
        test_result_saving_path="$working_dir/dominance_prediction/aa_subs/test_results.csv"

        if [ ! -f $test_result_saving_path ];
        then
            python baselines/dominance_aa_subs.py --train_fasta_path $train_fasta_path --test_fasta_path $testing_viruses_path --ckpt_saving_dir $ckpt_saving_dir --test_result_saving_path $test_result_saving_path --min_testing_time $min_testing_time --max_testing_time $max_testing_time --test
        fi
    done
done
