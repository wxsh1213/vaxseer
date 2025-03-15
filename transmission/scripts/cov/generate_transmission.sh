year=$1 # "2017"
month=$2 # "a_h3n2"
GPUS=$3 # "0,"
temperatures=$4

BATCH_SIZE="128"
generation_seq_number="500"
testing_window="3"

cd ../vaxseer

for temperature in $temperatures
do 
    for continent in "africa" "asia" "europe" "north_america" "oceania" "south_america"
    do
        echo $year-$month $GPUS $continent $temperature 
        
        root_dir="../runs/cov_continents/2019-12_to_$year-"$month"_1M/transmission"

        DEFAULT_ROOT_DIR="$root_dir/generations_"$generation_seq_number"_$testing_window"M"/$continent/temp_$temperature"

        ckpt_path=$(ls $root_dir/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)

        end_of_training=`expr \( $year - 2019 \) \* 12 + \( $month - 12 \) `
        min_testing_time=`expr $end_of_training + $testing_window`
        max_testing_time=`expr $min_testing_time + $testing_window - 1`

        echo "min_testing_time: $min_testing_time, max_testing_time: $max_testing_time"

        python -m bin.train \
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
            --set_data_properties "{\"location\": \"$continent\"}" \
            --do_sample true --temperature $temperature
    done
done