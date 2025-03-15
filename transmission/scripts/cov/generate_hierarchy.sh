year_and_month=$1
GPUS=$2
temperatures=$3
block_trans_model=$4 # continent_to_continent / country_to_continent

countries=( "africa/south_africa" "asia/china" "asia/india" "asia/indonesia" "asia/israel" "asia/japan" "asia/south_korea" "europe/austria" "europe/belgium" "europe/czech_republic" "europe/denmark" "europe/france" "europe/germany" "europe/ireland" "europe/italy" "europe/luxembourg" "europe/netherlands" "europe/norway" "europe/poland" "europe/russia" "europe/slovenia" "europe/spain" "europe/sweden" "europe/switzerland" "europe/turkey" "europe/united_kingdom" "north_america/canada" "north_america/mexico" "north_america/usa" "oceania/australia" "south_america/brazil" "south_america/peru")

continent2countries="../transmission/data/continent2countries/cov/top32_minCnt1000_agg_complete_3.json"

generation_seq_number="500"
testing_window="3"
arr=(${year_and_month//"-"/ })
year=${arr[0]}
month=${arr[1]}

cd ../vaxseer

for temperature in $temperatures
do
    for continent_and_country in ${countries[@]}
    do
        LR="1e-5"
        BATCH_SIZE="128"
        NUM_LAYER="6"

        arr=(${continent_and_country//"/"/ })
        continent=${arr[0]}
        country=${arr[1]}

        echo $year-$month $GPUS $continent/$country $temperature

        root_dir="../runs/cov_countries/2020-01_to_$year-"$month"_1M/transmission_hierachy/$block_trans_model/agg_complete_3"

        DEFAULT_ROOT_DIR="$root_dir/generations_"$generation_seq_number"_"$testing_window"M/"$continent"_"$country"/temp_$temperature"

        ckpt_path=$(ls $root_dir/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)

        end_of_training=`expr \( $year - 2020 \) \* 12 + \( $month - 1 \) `
        min_testing_time=`expr $end_of_training + $testing_window`
        max_testing_time=`expr $min_testing_time + $testing_window - 1`
        
        echo "min_testing_time: $min_testing_time, max_testing_time: $max_testing_time"

        echo $ckpt_path

        python -m bin.train \
            --data_module lm_weighted_location \
            --default_root_dir $DEFAULT_ROOT_DIR \
            --continent_to_country_mapping_file $continent2countries \
            --max_position_embeddings 1024 \
            --accelerator gpu \
            --devices $GPUS \
            --batch_size $BATCH_SIZE \
            --precision 16 \
            --num_workers 11 \
            --model gpt2_time_transmission_hierarchy \
            --transformer_offset \
            --data_properties continent location \
            --normalize_time_a 100 \
            --block_trans_model $block_trans_model \
            --implement_version 1 \
            --predict \
            --resume_from_checkpoint $ckpt_path \
            --max_testing_time $max_testing_time \
            --min_testing_time $min_testing_time \
            --generation_seq_number $generation_seq_number \
            --set_data_properties "{\"continent\": \"$continent\", \"location\": \"$continent_and_country\" }" \
            --do_sample true --temperature $temperature --remap_continent true
    done
done