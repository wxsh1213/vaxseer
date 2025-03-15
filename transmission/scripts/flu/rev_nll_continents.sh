year=$1
GPUS=$2
temperature=$3
seed=$4

cd ../vaxseer

for continent in "asia" "europe" "oceania" "south_america" "africa" "north_america"
do
    echo $year $continent $temperature

    index=`expr \( $year - 2018 \) \* 2 + 30`

    min_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 5`
    max_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 7`
    
    root_dir="../runs/flu_ah3n2_continents/2003-10_to_2018-02_2M/transmission/"
    DEFAULT_ROOT_DIR="$root_dir/test_rev_seed"$seed"/$continent/temp_$temperature"
    TEST_SET="$root_dir/generations_500/$continent/temp_$temperature/lightning_logs/version_0/predictions.fasta"

    RESUME_FROM_CHECKPOINT=$(ls ../runs/flu_ah3n2_continents/oracles/2003-10_to_2023-04_2M/$continent/seed_$seed/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)

    ls $TEST_SET
    if [ -f "$TEST_SET" ]; then
        python -m bin.train --default_root_dir $DEFAULT_ROOT_DIR --test_data_paths $TEST_SET --max_position_embeddings 1024 --accelerator gpu --devices $GPUS --batch_size 16 --precision 16 --strategy ddp --num_workers 11 --test --resume_from_checkpoint $RESUME_FROM_CHECKPOINT --model gpt2_time_new --min_testing_time $min_testing_time --max_testing_time $max_testing_time --temperature 1.0 
    fi    

done

