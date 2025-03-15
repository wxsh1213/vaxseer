year=$1 # "2018"
devices=$2 # "2,3"
data_root_dir="../data/gisaid/flu/ha_processed_continents"
output_dir="../runs/flu_ah3n2_continents"

continents=("africa" "asia" "europe" "north_america" "oceania" "south_america")

TRAIN_DATA="$data_root_dir/2003-10_to_"$year"-02_2M/a_h3n2/continents/human_minBinSize10_minLen553_"${continents[0]}"_location.fasta"
for loc in "${continents[@]:1}"
do
    TRAIN_DATA=$TRAIN_DATA" $data_root_dir/2003-10_to_"$year"-02_2M/a_h3n2/continents/human_minBinSize10_minLen553_"$loc"_location.fasta"
done

echo "Train data:" $TRAIN_DATA
echo "Year: $year, devices: $devices"

DEFAULT_ROOT_DIR="$output_dir/2003-10_to_$year-02_2M/transmission"

cd ../vaxseer

python -m bin.train \
    --data_module lm_weighted_location \
    --default_root_dir $DEFAULT_ROOT_DIR \
    --data_path $TRAIN_DATA \
    --max_position_embeddings 1024 \
    --max_epochs -1 --max_steps 80000 \
    --accelerator gpu \
    --devices $devices \
    --batch_size 16 \
    --num_hidden_layers 6 \
    --gradient_clip_val 2.0 \
    --num_workers 11 \
    --model gpt2_time_transmission \
    --learning_rate 1e-5 \
    --data_properties location \
    --weight_loss_by_count true \
    --normalize_time_a 100 \
    --transformer_offset \
    --pos_function "softplus" --offset_pos_func softmax \
    --precision 16 \
    --implement_version 1 --num_host 6 \
    --global_logits_reg_w 0.1 --strategy ddp