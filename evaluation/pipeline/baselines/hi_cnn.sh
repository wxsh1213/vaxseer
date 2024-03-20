device="4"
month="02"

ckpt_root_dir="../runs/flu_hi_cnn_regressor"

for year in `seq 2012 2014`
do
    for subtype in "a_h1n1" "a_h3n2"
    do
        hi_ckpt=$(ls $ckpt_root_dir/before_$year-$month/"$subtype"_seed=1005/random_split/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)

        echo $hi_ckpt

        year_minus_three=`(expr $year - 3)`
        index=`expr \( $year - 2018 \) \* 2 + 30`

        working_dir="../runs/pipeline/$year-"$month"/$subtype/vaccine_set=$year_minus_three-"$month"-$year-"$month"___virus_set=$year_minus_three-"$month"-$year-"$month""
        classifier_root_dir="$working_dir/vaccine_virus_pairs/prediction/cnn"
        hi_pred_path=$classifier_root_dir"/predictions.csv"
        pairs_save_path="$working_dir/vaccine_virus_pairs/pairs.csv"

        if [ ! -f $hi_pred_path ];
        then
            cd ../vaxseer

            python -m bin.train --default_root_dir $classifier_root_dir --data_module hi_regression_aln --model esm_regressor_cnn --accelerator gpu --devices $device, --batch_size 32 --num_workers 11 --precision 16 --predict --resume_from_checkpoint $hi_ckpt --predict_index_path $pairs_save_path --category false --vocab msa_transformer
            new_hi_pred_path=$classifier_root_dir"/lightning_logs/version_0/predictions.csv"
            
            mv $new_hi_pred_path $classifier_root_dir
            
            rm -r $classifier_root_dir"/lightning_logs"
        fi

    done
done
