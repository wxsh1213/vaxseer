year=$1 # e.g. 2021
month=$2 # e.g. 07
devices=$3 # e.g. "5,7"
block_trans_model=$4 # continent_to_continent / country_to_continent

data_root_dir="../data/gisaid/cov/spike_rbd_processed_countries"
output_dir="../runs/cov_countries"
continent2countries="../transmission/data/continent2countries/cov/top32_minCnt1000_agg_complete_3.json"
TRAIN_DATA="$data_root_dir/2020-01_to_"$year"-"$month"_1M/all/human_minBinSize100_minLen223_maxLen223_location_region2.fasta"
DEFAULT_ROOT_DIR="$output_dir/2020-01_to_"$year"-"$month"_1M/transmission_hierachy/$block_trans_model/agg_complete_3"

echo Year-month: $year-$month, devices: $devices 
echo Train data: $TRAIN_DATA

cd ../vaxseer

python -m bin.train \
    --data_module lm_weighted_location \
    --default_root_dir $DEFAULT_ROOT_DIR \
    --data_path $TRAIN_DATA \
    --max_position_embeddings 1024 \
    --max_epochs -1 --max_steps 30000 \
    --val_check_interval 300 --set_none_check_val_every_n_epoch \
    --accelerator gpu \
    --devices $devices \
    --batch_size 128 \
    --num_hidden_layers 6 \
    --gradient_clip_val 2.0 \
    --precision 16 \
    --num_workers 11 \
    --model gpt2_time_transmission_hierarchy \
    --learning_rate 5e-5 \
    --data_properties continent location \
    --normalize_time_a 100 \
    --block_trans_model $block_trans_model \
    --implement_version 1 \
    --continent_loss_weight 0.1 --reuse_transformer_for_cross_block_trans true \
    --use_simple_continent_model true \
    --pos_function "softplus" --cross_continent_reg 0.0 \
    --continent_to_country_mapping_file $continent2countries --remap_continent true --strategy ddp \
    --use_linear_init_trans_weight true --add_other_countries false