devices=$1 # 1,
location=$2 # "asia"
random_seed=$3 # 1005
data_root_dir="../data/gisaid/flu/ha_processed_continents"
output_dir="../runs/flu_ah3n2_continents/oracles"

TRAIN_DATA="$data_root_dir/2003-10_to_2023-04_2M/a_h3n2/continents/human_minBinSize10_minLen553_"$location"_location.fasta"

echo "Location:" $location

DEFAULT_ROOT_DIR="$output_dir/2003-10_to_2023-04_2M/$location/seed_$random_seed"

cd ../vaxseer

python -m bin.train \
    --default_root_dir $DEFAULT_ROOT_DIR \
    --data_path $TRAIN_DATA --max_position_embeddings 1024 \
    --max_epochs -1 --max_steps 80000 --accelerator gpu --devices $devices \
    --batch_size 32 --num_hidden_layers 6 \
    --gradient_clip_val 2.0 --precision 16 --num_workers 11 \
    --model gpt2_time_new --learning_rate 1e-5 \
    --weight_loss_by_count true --transformer_offset --seed $random_seed