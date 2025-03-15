

year=$1 # "2017"
month=$2 # "a_h3n2"
GPUS=$3 # "0,"

temperature="1.0"
generation_seq_number="500"
testing_window="3"

cd ../vaxseer

for continent in "africa" "asia" "europe" "north_america" "oceania" "south_america"  # 
do
    echo $year $GPUS $continent

    LR="1e-5"
    BATCH_SIZE="1"
    NUM_LAYER="6"

    end_of_training=`expr \( $year - 2019 \) \* 12 + \( $month - 12 \) `

    root_dir="../runs/cov_continents/2019-12_to_$year-"$month"_1M/transmission"
    min_testing_time=`expr $end_of_training`
    max_testing_time=`expr $min_testing_time + $testing_window - 1`

    DEFAULT_ROOT_DIR="$root_dir/generations_beam_search_"$generation_seq_number"_$testing_window"M"/$continent/temp_$temperature"
    ckpt_path=$(ls $root_dir/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)

    echo "min_testing_time: $min_testing_time, max_testing_time: $max_testing_time"

    python -m bin.train \
        --default_root_dir $DEFAULT_ROOT_DIR \
        --max_position_embeddings 1024 \
        --max_epochs 100 \
        --accelerator gpu \
        --devices $GPUS \
        --batch_size $BATCH_SIZE \
        --num_hidden_layers $NUM_LAYER \
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
        --generation_seq_number 1 \
        --set_data_properties "{\"location\": \"$continent\" }" \
        --do_sample false --temperature $temperature --num_beams $generation_seq_number
done