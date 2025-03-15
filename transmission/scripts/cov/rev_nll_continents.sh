testing_window="3"

year=$1 # "2021"
month=$2 # "10"
GPUS=$3
temperature=$4
oracle_model=$5 # 1 2 3

cd ../vaxseer

for continent in "asia" "europe" "oceania" "africa" "north_america" "south_america"
do
    echo $year-$month $GPUS $temperature

    end_of_training=`expr \( $year - 2019 \) \* 12 + \( $month - 12 \) `
    min_testing_time=`expr $end_of_training + $testing_window`
    max_testing_time=`expr $min_testing_time + $testing_window - 1`

    echo "min_testing_time: $min_testing_time, max_testing_time: $max_testing_time"
    
    ######## set-up oracle models
    case "$oracle_model" in
        1)
            RESUME_FROM_CHECKPOINT=$(ls ../runs/cov_continents/oracles/2019-12_to_2023-12_1M/$continent/1/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)
            test_rev_dir="test_rev_1"
            ;;
        2)
            RESUME_FROM_CHECKPOINT=$(ls ../runs/cov_continents/oracles/2019-12_to_2023-12_1M/$continent/2/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)
            test_rev_dir="test_rev_2"
            ;;
        3)
            RESUME_FROM_CHECKPOINT=../runs/cov_continents/oracles/2019-12_to_2023-12_1M/$continent/3/lightning_logs/version_0/checkpoints/last.ckpt
            test_rev_dir="test_rev_3"
            ;;
    esac
    echo "test_rev_dir: $test_rev_dir, checkpoint: $RESUME_FROM_CHECKPOINT"
    ######## set-up oracle models

    # # Transimission model V1
    MODEL_DIR="../runs/cov_continents/2019-12_to_"$year"-"$month"_1M/transmission/"

    DEFAULT_ROOT_DIR="$MODEL_DIR/$test_rev_dir/$continent/temp_$temperature"
    TEST_SET="$MODEL_DIR/generations_500_"$testing_window"M/$continent/temp_$temperature/lightning_logs/version_0/predictions.fasta"

    ls $TEST_SET

    if [ -f "$TEST_SET" ]; then
        python -m bin.train --default_root_dir $DEFAULT_ROOT_DIR --test_data_paths $TEST_SET --max_position_embeddings 1024 --accelerator gpu --devices $GPUS --batch_size 64 --precision 16 --strategy ddp --num_workers 11 --test --resume_from_checkpoint $RESUME_FROM_CHECKPOINT --model gpt2_time_new --min_testing_time $min_testing_time --max_testing_time $max_testing_time 
    fi

done