year=$1 # "2018"
devices=$2 # "2,"
block_trans_model=$3 # continent_to_continent / country_to_continent

data_root_dir="../data/gisaid/flu/ha_processed_countries"
output_dir="../runs/flu_ah3n2_countries"

continent_to_country_mapping_file="../transmission/data/continent2countries/flu/top16_minCnt1000_agg_complete_2.json"
TRAIN_DATA="$data_root_dir/2003-10_to_"$year"-02_2M/a_h3n2/human_minBinSize10_minLen553_maxLen566_location_region2.fasta"

echo "Train data:" $TRAIN_DATA
echo "Year: $year, devices: $devices"

DEFAULT_ROOT_DIR="$output_dir/2003-10_to_$year-02_2M/transmission_hierachy/$block_trans_model/agg_complete_2/"

cd ../vaxseer

python -m bin.train \
        --data_module lm_weighted_location \
        --default_root_dir $DEFAULT_ROOT_DIR \
        --data_path $TRAIN_DATA \
        --max_position_embeddings 1024 \
        --max_epochs -1 --max_steps 80000 \
        --val_check_interval 400 --set_none_check_val_every_n_epoch \
        --accelerator gpu \
        --devices $devices \
        --batch_size 32 \
        --num_hidden_layers 6 \
        --gradient_clip_val 2.0 \
        --precision 16 \
        --num_workers 11 \
        --model gpt2_time_transmission_hierarchy \
        --learning_rate 1e-5 \
        --data_properties continent country \
        --add_other_countries false \
        --normalize_time_a 100 \
        --block_trans_model $block_trans_model \
        --implement_version 1 \
        --continent_loss_weight 0.1 --reuse_transformer_for_cross_block_trans true \
        --use_simple_continent_model true \
        --continent_share_base_models true \
        --remap_continent true --continent_to_country_mapping_file $continent_to_country_mapping_file \
        --cross_continent_reg 0.0 --use_linear_init_trans_weight true