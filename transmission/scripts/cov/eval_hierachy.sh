year=$1 # '2022'
month=$2 # '01'
GPUS="$3"
block_trans_model=$4 # country_to_continent  /  continent_to_continent

temperature_array=(0.2 0.4 0.6 0.8 1.0)
testing_window="3"
continent2countries="../transmission/data/continent2countries/cov/top32_minCnt1000_agg_complete_3.json"

cd ../vaxseer

for i in 0 1 2 3 4;
do 
    temperature=${temperature_array[$i]}

    echo $year-$month $GPUS $temperature

    root_dir="../runs/cov_countries/2020-01_to_"$year"-"$month"_1M/transmission_hierachy/$block_trans_model/agg_complete_3"

    DEFAULT_ROOT_DIR="$root_dir/test_"$testing_window"M/temp_$temperature"
    ckpt_path=$(ls $root_dir/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)

    end_of_training=`expr \( $year - 2020 \) \* 12 + \( $month - 1 \) `
    min_testing_time=`expr $end_of_training + $testing_window`
    max_testing_time=`expr $min_testing_time + $testing_window - 1`

    echo $min_testing_time $max_testing_time

    index=`expr \( $year - 2020 \) \* 12 \/ $testing_window + \( $month + $testing_window - 1 \) \/ $testing_window `

    TEST_DATA="../data/gisaid/cov/spike_rbd_processed_countries/2020-01_to_2023-12_"$testing_window"M/all/human_minBinSize100_minLen223_maxLen223_location_region2_bins/$index.fasta"

    echo $TEST_DATA
    
    python -m bin.train \
        --data_module lm_weighted_location \
        --default_root_dir $DEFAULT_ROOT_DIR \
        --test_data_paths $TEST_DATA \
        --continent_to_country_mapping_file $continent2countries \
        --max_position_embeddings 1024 \
        --accelerator gpu \
        --devices $GPUS \
        --batch_size 64 \
        --precision 16 \
        --strategy ddp \
        --num_workers 11 \
        --test \
        --resume_from_checkpoint $ckpt_path \
        --model gpt2_time_transmission_hierarchy \
        --data_properties continent location \
        --min_testing_time $min_testing_time \
        --max_testing_time $max_testing_time \
        --temperature $temperature --remap_continent true 
done