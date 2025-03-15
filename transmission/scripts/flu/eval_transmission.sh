year=$1 # "2015"
devices=$2
temperature=$3 # 0.2

BATCH_SIZE="16"

model_root_dir="../runs/flu_ah3n2_continents/2003-10_to_"$year"-02_2M/transmission"

cd ../vaxseer

for continent in "asia" "europe" "oceania" "south_america" "africa" "north_america"
do
    echo Year: $year, devices: $devices, temperature: $temperature, continent: $continent

    max_testing_time=`expr \( $year - 2004 \) \* 6 + 8`
    min_testing_time=`expr \( $year - 2004 \) \* 6 + 6`
    index=`expr \( $year - 2018 \) \* 2 + 30`

    DEFAULT_ROOT_DIR="$model_root_dir/test/$continent/temp_$temperature"
    ckpt_path=$(ls $model_root_dir/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)

    test_data_paths="../data/gisaid/flu/ha_processed_continents/2003-10_to_2023-04_6M/a_h3n2/continents/human_minBinSize10_minLen553_"$continent"_location_bins/$index.fasta"

    python -m bin.train \
        --data_module lm_weighted_location \
        --default_root_dir $DEFAULT_ROOT_DIR \
        --test_data_paths $test_data_paths \
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

