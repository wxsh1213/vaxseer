#!/bin/bash

VALID_ARGS=$(getopt -o c:t:d:h:p:a:b:e:f:g: --long candidate_vaccine_path:,testing_viruses_path:,working_directory:,hi_predictor_ckpt:,domiance_predictor_ckpt:,devices:,min_testing_time:,max_testing_time: -- "$@")

if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    -c | --candidate_vaccine_path)
        candidate_vaccine_path=$2
        echo "candidate_vaccine_path: $candidate_vaccine_path"
        shift 2
        ;;
    -t | --testing_viruses_path)
        testing_viruses_path=$2
        echo "testing_viruses_path: $testing_viruses_path"
        shift 2
        ;;
    -d | --working_directory)
        working_directory=$2
        echo "working_directory: $working_directory"
        shift 2
        ;;
    -h | --hi_predictor_ckpt)
        hi_predictor_ckpt=$2
        echo "hi_predictor_ckpt: $hi_predictor_ckpt"
        shift 2
        ;;
    -p | --domiance_predictor_ckpt)
        domiance_predictor_ckpt=$2
        echo "domiance_predictor_ckpt: $domiance_predictor_ckpt"
        shift 2
        ;;
    -e | --devices)
        devices=$2
        echo "devices: $devices"
        shift 2
        ;;
    -f | --min_testing_time)
        min_testing_time=$2
        echo "min_testing_time: $min_testing_time"
        shift 2
        ;;
    -g | --max_testing_time)
        max_testing_time=$2
        echo "max_testing_time: $max_testing_time"
        shift 2
        ;;
    --) shift; 
        break 
        ;;
  esac
done


# 1. Run mmseqs to build vaccine-virus pairs
mmseqs_save_dir="$working_directory/vaccine_virus_pairs"
mkdir -p $mmseqs_save_dir
mmseqs_save_path="$mmseqs_save_dir/align.m8"

if [ ! -f $mmseqs_save_path ]; 
then 
    mmseqs easy-search $testing_viruses_path $candidate_vaccine_path $mmseqs_save_path tmp --format-output "query,target,qaln,taln,qstart,qend,tstart,tend,mismatch" --max-seqs 2000
fi

cd "../data/"
pairs_save_path="$working_directory/vaccine_virus_pairs/pairs.csv"

if [ ! -f $pairs_save_path ];
then
    python -m antigenicity.build_test_pairs --alignment_path $mmseqs_save_path --virus_fasta_path $testing_viruses_path --vaccine_fasta_path $candidate_vaccine_path --save_path $pairs_save_path
fi

# 2. Predict the hi values for pairs
cd ../vaxseer

classifier_root_dir="$working_directory/vaccine_virus_pairs/prediction/max_steps_150k/"
hi_pred_path=$classifier_root_dir"/predictions.csv"

if [ ! -f $hi_pred_path ];
then
    python -m bin.train --default_root_dir $classifier_root_dir --data_module hi_regression_aln --model esm_regressor --accelerator gpu --devices $devices --batch_size 64 --learning_rate 3e-4 --num_workers 36 --precision 16 --max_epochs 100 --predict --resume_from_checkpoint $hi_predictor_ckpt --predict_index_path $pairs_save_path --category false
fi

# 3. Predict the domiance for viruses
lm_root_dir="$working_directory/dominance_prediction"
prob_pred_path=$lm_root_dir"/lightning_logs/version_0/test_results.csv"
if [ ! -f $prob_pred_path ];
then
    python -m bin.train --default_root_dir $lm_root_dir --test_data_paths $testing_viruses_path --max_position_embeddings 1024 --accelerator gpu --devices $devices --batch_size 64 --precision 16 --strategy ddp --num_workers 11 --test --resume_from_checkpoint $domiance_predictor_ckpt --model gpt2_time_new --max_testing_time $max_testing_time --min_testing_time $min_testing_time
    cp $classifier_root_dir"/lightning_logs/version_0/predictions.csv" $hi_pred_path
    rm -r $classifier_root_dir"/lightning_logs"
fi


