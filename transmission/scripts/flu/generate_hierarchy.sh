year=$1
GPUS=$2
temperatures=$3
block_trans_model=$4 # "continent_to_continent" # continent_to_continent / country_to_continent

generation_seq_number="500" # 
BATCH_SIZE="64"

countries=( europe/france europe/germany europe/russian_federation europe/spain europe/united_kingdom asia/china asia/japan north_america/canada north_america/united_states oceania/australia )

continent_to_country_mapping_file="../transmission/data/continent2countries/flu/top16_minCnt1000_agg_complete_2.json"

cd ../vaxseer

for temperature in $temperatures
do
    for continent_and_country in ${countries[@]}
    do
        arr=(${continent_and_country//"/"/ })
        continent=${arr[0]}
        country=${arr[1]}

        echo $year $GPUS $continent/$country $temperature
        
        root_dir="../runs/flu_ah3n2_countries/2003-10_to_$year-02_2M/transmission_hierachy/$block_trans_model/agg_complete_2"

        DEFAULT_ROOT_DIR="$root_dir/generations_$generation_seq_number/"$continent"_"$country"/temp_$temperature"

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
            --model gpt2_time_transmission_hierarchy \
            --transformer_offset \
            --data_properties continent country \
            --normalize_time_a 100 \
            --block_trans_model $block_trans_model \
            --implement_version 1 \
            --predict \
            --resume_from_checkpoint $ckpt_path \
            --max_testing_time $max_testing_time \
            --min_testing_time $min_testing_time \
            --generation_seq_number $generation_seq_number \
            --set_data_properties "{\"continent\": \"$continent\", \"country\": \"$country\" }" \
            --do_sample true --temperature $temperature \
            --continent_to_country_mapping_file $continent_to_country_mapping_file --remap_continent true
    done
done