
testing_window="3"

year=$1
month=$2
temperature=$3
GPUS=$4
oracle_model=$5
block_trans_model=$6 # continent_to_continent / country_to_continent


# 32 countries
countries=( africa/south_africa asia/china asia/india asia/indonesia asia/israel asia/japan asia/south_korea europe/austria europe/belgium europe/czech_republic europe/denmark europe/france europe/germany europe/ireland europe/italy europe/luxembourg europe/netherlands europe/norway europe/poland europe/russia europe/slovenia europe/spain europe/sweden europe/switzerland europe/turkey europe/united_kingdom north_america/canada north_america/mexico north_america/usa oceania/australia south_america/brazil south_america/peru )

cd ../vaxseer

for continent_and_country in ${countries[@]}
do
    arr=(${continent_and_country//"/"/ })
    continent=${arr[0]}
    country=${arr[1]}

    echo $year-$month $GPUS $temperature $continent/$country

    end_of_training=`expr \( $year - 2019 \) \* 12 + \( $month - 12 \) `
    min_testing_time=`expr $end_of_training + $testing_window`
    max_testing_time=`expr $min_testing_time + $testing_window - 1`

    echo "min_testing_time: $min_testing_time, max_testing_time: $max_testing_time"

    ######## set-up oracle models
    case "$oracle_model" in
        1)
            RESUME_FROM_CHECKPOINT=$(ls ../runs/cov_countries/oracles/2019-12_to_2023-12_1M/"$continent"/"$country"/1/epoch=*-step=*.ckpt)
            test_rev_dir="test_rev_1"
            ;;
        2)
            RESUME_FROM_CHECKPOINT=$(ls ../runs/cov_countries/oracles/2019-12_to_2023-12_1M/"$continent"/"$country"/2/epoch=*-step=*.ckpt)
            test_rev_dir="test_rev_2"
            ;;
        3)
            RESUME_FROM_CHECKPOINT=$(ls ../runs/cov_countries/oracles/2019-12_to_2023-12_1M/"$continent"/"$country"/3/epoch=*-step=*.ckpt)
            test_rev_dir="test_rev_3"
            ;;
    esac
    echo "test_rev_dir: $test_rev_dir, checkpoint: $RESUME_FROM_CHECKPOINT"
    ######## set-up oracle models
    
    MODEL_DIR="../runs/cov_countries/2020-01_to_"$year"-"$month"_1M/transmission_hierachy/$block_trans_model/agg_complete_3/generations_500_3M/"
    DEFAULT_ROOT_DIR="$MODEL_DIR/$test_rev_dir/"$continent"_"$country"/temp_$temperature"
    TEST_SET="$MODEL_DIR/"$continent"_"$country"/temp_$temperature/lightning_logs/version_0/predictions.fasta"

    ls $TEST_SET
    if [ -f "$TEST_SET" ]; then
        python -m bin.train --default_root_dir $DEFAULT_ROOT_DIR --test_data_paths $TEST_SET --max_position_embeddings 1024 --accelerator gpu --devices $GPUS --batch_size 64 --precision 16 --strategy ddp --num_workers 11 --test --resume_from_checkpoint $RESUME_FROM_CHECKPOINT --model gpt2_time_new --min_testing_time $min_testing_time --max_testing_time $max_testing_time --temperature 1.0
    fi
done
