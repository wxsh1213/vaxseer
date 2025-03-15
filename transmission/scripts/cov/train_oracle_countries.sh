devices=$1
random_seed=$2

countries=( africa/south_africa asia/china asia/india asia/indonesia asia/israel asia/japan asia/south_korea europe/austria europe/belgium europe/czech_republic europe/denmark europe/france europe/germany europe/ireland europe/italy europe/luxembourg europe/netherlands europe/norway europe/poland europe/russia europe/slovenia europe/spain europe/sweden europe/switzerland europe/turkey europe/united_kingdom north_america/canada north_america/mexico north_america/usa oceania/australia south_america/brazil south_america/peru )


LR="3e-4"
BATCH_SIZE="256"
NUM_LAYER="6"
max_steps="20000"
oracle_name="2"

# setting1:
# --weight_loss_by_count false --max_steps 30000 --learning_rate 1e-4
# setting2: 
# --weight_loss_by_count true --max_steps 20000 --learning_rate 3e-4 --seed 1212
# setting3:
# --weight_loss_by_count true --max_steps 20000 --learning_rate 3e-4 --seed 1005

cd ../vaxseer

for continent_and_country in "${countries[@]}";
do
    arr=(${continent_and_country//"/"/ })
    continent=${arr[0]}
    country=${arr[1]}
    
    echo "Training:" $continent/$country

    TRAIN_DATA="../data/gisaid/cov/spike_rbd_processed_countries/2019-12_to_2023-12_1M/all/human_minBinSize100_minLen223_maxLen223_location_region2_locations/$continent/$country.fasta"

    DEFAULT_ROOT_DIR="../runs/cov_countries/oracles/2019-12_to_2023-12_1M/$continent/$country/$oracle_name"

    python -m bin.train --default_root_dir $DEFAULT_ROOT_DIR --data_path $TRAIN_DATA --max_position_embeddings 1024 --max_epochs -1 --max_steps $max_steps --accelerator gpu --devices $devices --batch_size $BATCH_SIZE --num_hidden_layers $NUM_LAYER --gradient_clip_val 2.0 --precision 16 --num_workers 11 --model gpt2_time_new --learning_rate $LR --weight_loss_by_count true --val_check_interval 300 --set_none_check_val_every_n_epoch --seed $random_seed 

done



