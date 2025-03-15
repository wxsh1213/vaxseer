devices=$1 # "1,"
seed=$2 # "529"

countries=( europe/france europe/germany europe/russian_federation europe/spain europe/united_kingdom asia/china asia/japan north_america/canada north_america/united_states oceania/australia )

cd ../vaxseer

for i in `seq 0 9`;
do
    continent_and_country=${countries[$i]}
    arr=(${continent_and_country//"/"/ })
    continent=${arr[0]}
    country=${arr[1]}

    TRAIN_DATA="../data/gisaid/flu/ha_processed_countries/2003-10_to_2023-04_6M/a_h3n2/human_minBinSize10_minLen553_maxLen566_location_region2_locations/$continent/$country.fasta"

    echo "Training" $continent $country    

    DEFAULT_ROOT_DIR="../runs/flu_ah3n2_countries/oracles/2003-10_to_2023-04_6M/seed_$seed/$continent/$country"
    
    python -m bin.train --default_root_dir $DEFAULT_ROOT_DIR \
        --data_path $TRAIN_DATA --max_position_embeddings 1024 \
        --max_epochs -1 --max_steps 20000 \
        --accelerator gpu --devices $devices --batch_size 32 \
        --num_hidden_layers 6 --gradient_clip_val 2.0 --precision 16 \
        --num_workers 11 --model gpt2_time_new --learning_rate 1e-5  \
        --weight_loss_by_count true --transformer_offset \
        --val_check_interval 200 --set_none_check_val_every_n_epoch --seed $seed
done