year=$1 # e.g. 2021
month=$2 # e.g. 07
devices=$3 # e.g. "2,3,4,5"

data_root_dir="../data/gisaid/cov/spike_rbd_processed_continents"
output_dir="../runs/cov_continents"

continents=("africa" "asia" "europe" "north_america" "oceania" "south_america")

TRAIN_DATA="$data_root_dir/2019-12_to_"$year"-"$month"_1M/all/continents/human_minBinSize100_minLen223_maxLen223_"${continents[0]}"_location.fasta"
for loc in "${continents[@]:1}"
do
    TRAIN_DATA=$TRAIN_DATA" $data_root_dir/2019-12_to_"$year"-"$month"_1M/all/continents/human_minBinSize100_minLen223_maxLen223_"$loc"_location.fasta"
done

echo Train data: $TRAIN_DATA
echo Year-month: $year-$month, devices: $devices 

DEFAULT_ROOT_DIR="$output_dir/2020-01_to_"$year"-"$month"_1M/transmission"

cd ../vaxseer

python -m bin.train \
    --default_root_dir $DEFAULT_ROOT_DIR \
    --data_path $TRAIN_DATA \
    --max_position_embeddings 1024 \
    --max_epochs -1 --max_steps 30000 \
    --accelerator gpu \
    --devices $devices \
    --batch_size 64 \
    --num_hidden_layers 6 \
    --gradient_clip_val 2.0 \
    --precision 16 \
    --num_workers 11 \
    --model gpt2_time_transmission \
    --learning_rate 5e-5 \
    --data_properties location \
    --normalize_time_a 100 \
    --pos_function softplus \
    --offset_pos_function softmax \
    --weight_loss_by_count true \
    --implement_version 1 --num_host 6 --transformer_offset \
    --global_logits_reg_w 0.1 \
    --strategy ddp

