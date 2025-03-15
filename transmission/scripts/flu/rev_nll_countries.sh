year=$1 # 2019
GPUS=$2
temperature=$3
oracle_model=$4

countries=( asia/china asia/japan europe/france europe/germany europe/russian_federation europe/spain europe/united_kingdom north_america/canada north_america/united_states oceania/australia )

cd ../vaxseer

for location in ${countries[@]};
do  
    arr=(${location//"/"/ })
    continent=${arr[0]}
    country=${arr[1]}

    ######## set-up oracle models
    case "$oracle_model" in
        1)
            RESUME_FROM_CHECKPOINT=$(ls ../runs/flu_ah3n2_countries/oracles/2003-10_to_2023-04_6M/seed_1005/$continent/$country/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)
            test_rev_dir="test_rev_seed_1005"
            ;;
        2)
            RESUME_FROM_CHECKPOINT=$(ls ../runs/flu_ah3n2_countries/oracles/2003-10_to_2023-04_6M/seed_1213/$continent/$country/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)
            test_rev_dir="test_rev_seed_1213"
            ;;
        3)
            RESUME_FROM_CHECKPOINT=$(ls ../runs/flu_ah3n2_countries/oracles/2003-10_to_2023-04_6M/seed_529/$continent/$country/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)
            test_rev_dir="test_rev_seed_529"
            ;;
    esac
    echo "test_rev_dir: $test_rev_dir, checkpoint: $RESUME_FROM_CHECKPOINT"
    ######## set-up oracle models

    echo $year $GPUS $subtype $temperature $continent/$country

    index=`expr \( $year - 2018 \) \* 2 + 30`

    min_testing_time=$index
    max_testing_time=$index

    root_dir="../runs/flu_ah3n2_countries/2003-10_to_"$year"-02_2M/transmission_hierachy/continent_to_continent/agg_complete_2/generations_500"
    
    DEFAULT_ROOT_DIR="$root_dir/$test_rev_dir/"$continent"_"$country"/temp_$temperature"
    TEST_SET="$root_dir/"$continent"_"$country"/temp_$temperature/lightning_logs/version_0/predictions.fasta"

    ls $TEST_SET
    if [ -f "$TEST_SET" ]; then
        python -m bin.train --default_root_dir $DEFAULT_ROOT_DIR --test_data_paths $TEST_SET --max_position_embeddings 1024 --accelerator gpu --devices $GPUS --batch_size 32 --precision 16 --strategy ddp --num_workers 11 --test --resume_from_checkpoint $RESUME_FROM_CHECKPOINT --model gpt2_time_new --min_testing_time $min_testing_time --max_testing_time $max_testing_time --temperature 1.0
    fi    

done