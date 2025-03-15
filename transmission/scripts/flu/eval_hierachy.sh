
year=$1 # "2017"
devices=$2 # "5,"
temperature="$3" # 0.2
block_trans_model="$4"

continent_to_country_mapping_file="../transmission/data/continent2countries/flu/top16_minCnt1000_agg_complete_2.json"

echo $year $GPUS $temperature

BATCH_SIZE="16"

model_root_dir="../runs/flu_ah3n2_countries/2003-10_to_"$year"-02_2M/transmission_hierachy/$block_trans_model/agg_complete_2/"

DEFAULT_ROOT_DIR="$model_root_dir/test/temp_$temperature"
ckpt_path=$(ls $model_root_dir/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)

max_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 7`
min_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 5`
index=`expr \( $year - 2018 \) \* 2 + 30`

test_data_paths="../data/gisaid/flu/ha_processed_countries/2003-10_to_2023-04_6M/a_h3n2/human_minBinSize10_minLen553_maxLen566_location_region2_bins/$index.fasta"

cd ../vaxseer

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
    --model gpt2_time_transmission_hierarchy \
    --transformer_offset \
    --data_properties continent country \
    --normalize_time_a 100 \
    --implement_version 1 \
    --block_trans_model $block_trans_model \
    --test \
    --resume_from_checkpoint $ckpt_path \
    --max_testing_time $max_testing_time \
    --min_testing_time $min_testing_time \
    --continent_to_country_mapping_file $continent_to_country_mapping_file --remap_continent true \
    --temperature $temperature 