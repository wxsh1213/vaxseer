year="$1"
subtype="$2"
gpu="$3"
dir_ckpt="$4" # "../runs/flu_lm/"

echo "Training dominance predictor for $year"

TRAIN_DATA="../data/gisaid/ha_processed/2003-10_to_"$year"-02_2M/$subtype/human_minBinSize100_lenQuantile0.2.fasta"

DEFAULT_ROOT_DIR="$dir_ckpt/2003-10_to_"$year"-02_2M/$subtype/human_minBinSize100_lenQuantile0.2/weight_loss_by_count"

nohup python -m bin.train \
    --default_root_dir $DEFAULT_ROOT_DIR \
    --data_path $TRAIN_DATA \
    --max_position_embeddings 1024 \
    --max_epochs 100 \
    --accelerator gpu \
    --devices $gpu, \
    --batch_size 16 \
    --num_hidden_layers 12 \
    --gradient_clip_val 2.0 \
    --precision 16 \
    --num_workers 11 \
    --model gpt2_time_new \
    --transformer_offset \
    --weight_loss_by_count true \
    --learning_rate "1e-5" > nohup.train_dominance_predictor.$subtype.$year.log 2>&1 &