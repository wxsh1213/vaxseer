year=$1 # "2021"
month=$2 # "07"
devices=$3

cd ../vaxseer

testing_window="3"
BATCH_SIZE="128"

for temperature in "0.2" "0.4" "0.6" "0.8" "1.0"
do
    echo $year-$month $devices $temperature

    end_of_training=`expr \( $year - 2019 \) \* 12 + \( $month - 12 \) `
    min_testing_time=`expr $end_of_training + $testing_window`
    max_testing_time=`expr $min_testing_time + $testing_window - 1`
    echo "min_testing_time: $min_testing_time, max_testing_time: $max_testing_time"

    index=`expr \( $year - 2020 \) \* 12 \/ $testing_window + \( $month + $testing_window - 1 \) \/ $testing_window `
    
    TEST_DATA="../data/gisaid/cov/spike_rbd_processed_continents/2020-01_to_2023-12_"$testing_window"M/all/human_minBinSize100_minLen223_maxLen223_location_region1_bins/$index.fasta"
    echo $TEST_DATA

    root_dir="../runs/cov_continents/2019-12_to_"$year"-"$month"_1M/transmission"
    DEFAULT_ROOT_DIR="$root_dir/test_"$testing_window"M/temp_$temperature"
    ckpt_path=$(ls $root_dir/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)
    
    python -m bin.train \
        --default_root_dir $DEFAULT_ROOT_DIR \
        --test_data_paths $TEST_DATA \
        --max_position_embeddings 1024 \
        --max_epochs 100 \
        --accelerator gpu \
        --devices $devices \
        --batch_size $BATCH_SIZE \
        --num_hidden_layers 6 \
        --gradient_clip_val 2.0 \
        --precision 16 \
        --num_workers 11 \
        --model gpt2_time_transmission \
        --transformer_offset \
        --data_properties location \
        --normalize_time_a 100 \
        --implement_version 1 \
        --test \
        --resume_from_checkpoint $ckpt_path \
        --max_testing_time $max_testing_time \
        --min_testing_time $min_testing_time \
        --temperature $temperature
        
done