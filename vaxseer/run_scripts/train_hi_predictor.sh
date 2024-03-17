year=$1 # "2018"
subtype=$2 # a_h3n2
gpu=$3 # 0
dir_ckpt="$4"

echo $year $subtype $gpu

split_method="random_split"
seed="1005"
month="02"

root_dir="$dir_ckpt/before_$year-$month/$subtype""_seed=$seed/random_split/max_steps_150k"
train_index_path="../data/antigenicity/hi_processed/before_$year-$month/"$subtype"_seed=$seed/random_split/train.csv"
valid_index_path="../data/antigenicity/hi_processed/before_$year-$month/"$subtype"_seed=$seed/random_split/valid.csv"
data_path="../data/gisaid/flu/ha.fasta"

nohup python -m bin.train \
    --default_root_dir $root_dir \
    --data_module hi_regression_aln \
    --model esm_regressor \
    --accelerator gpu \
    --devices $gpu, \
    --batch_size 32 \
    --learning_rate "1e-5" \
    --num_workers 11 \
    --precision 16 \
    --max_epochs -1 \
    --max_steps 150000 \
    --train_index_path $train_index_path \
    --valid_index_path $valid_index_path \
    --repr_layer 12 \
    --category false \
    --n_layers 12 \
    --model_name_or_path models/esm_msa1b_t12_100M_UR50S_args.pkl > nohup.train_hi_predictor.max_steps_150k.$subtype.$year.log 2>&1 &
