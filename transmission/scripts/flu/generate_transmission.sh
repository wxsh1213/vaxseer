year=$1 # "2017"
GPUS=$2 # "0,"
temperatures=$3

BATCH_SIZE="128"
generation_seq_number="500"

cd ../vaxseer

for temperature in $temperatures
do
    for continent in "africa" "asia" "north_america" "oceania" "south_america" "europe"
    do
        echo $year $GPUS $continent $temperature 

        root_dir="../runs/flu_ah3n2_continents/2003-10_to_"$year"-02_2M/transmission"
        DEFAULT_ROOT_DIR="$root_dir/generations_$generation_seq_number/$continent/temp_$temperature"
        ckpt_path=$(ls $root_dir/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)

        max_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 7`
        min_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 5`

        python -m bin.train \
            --data_module lm_weighted_location \
            --default_root_dir $DEFAULT_ROOT_DIR \
            --max_position_embeddings 1024 \
            --max_epochs 100 \
            --accelerator gpu \
            --devices $GPUS \
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
            --predict \
            --resume_from_checkpoint $ckpt_path \
            --max_testing_time $max_testing_time \
            --min_testing_time $min_testing_time \
            --generation_seq_number $generation_seq_number \
            --set_data_properties "{\"location\": \"$continent\" }" \
            --do_sample true --temperature $temperature --global_logits_reg_w 0.1
    done
done