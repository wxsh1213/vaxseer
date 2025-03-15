continent=$1 # "oceania"
devices="$2"

TRAIN_DATA="../data/gisaid/cov/spike_rbd_processed_continents/2019-12_to_2023-12_1M/all/continents/human_minBinSize100_minLen223_maxLen223_"$continent"_location.fasta"

echo $TRAIN_DATA

DEFAULT_ROOT_DIR="../runs/cov_continents/oracles/2019-12_to_2023-12_1M/$continent/2"

# three settings of oracle models:
# 1. --max_steps "100000" --weight_loss_by_count false
# 3. --transformer_offset --batch_size 160
# 2:

cd ../vaxseer

python -m bin.train --default_root_dir $DEFAULT_ROOT_DIR --data_path $TRAIN_DATA --max_position_embeddings 1024 --max_epochs -1 --max_steps "30000" --accelerator gpu --devices $devices --batch_size 256 --num_hidden_layers 6 --gradient_clip_val 2.0 --precision 16 --num_workers 11 --model gpt2_time_new --learning_rate 1e-4  --normalize_time_a 100 --weight_loss_by_count true