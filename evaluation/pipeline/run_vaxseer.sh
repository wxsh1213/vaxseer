
subtype=$1
year=$2
device=$3
month="02"

# ckpt_root_dir="runs" # directory of your checkpoints
# ckpt_root_dir="/Mounts/rbg-storage1/users/wenxian/devo_lightning/release_checkpoints/" # TODO: delete later
ckpt_root_dir="/Mounts/rbg-storage1/users/wenxian/devo_lightning/run_vaxseer/" # TODO: delete later

# V0
# lm_ckpt=$(ls $ckpt_root_dir/flu_lm/2003-10_to_"$year"-"$month"_2M/$subtype/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt) 
# hi_ckpt=$(ls $ckpt_root_dir/flu_hi_msa_regressor/before_$year-$month/"$subtype"_seed=1005/random_split/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)

# V1: training longer
lm_ckpt=$(ls $ckpt_root_dir/flu_lm/2003-10_to_"$year"-"$month"_2M/$subtype/human_minBinSize100_lenQuantile0.2/max_steps_100k/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt) 
hi_ckpt=$(ls $ckpt_root_dir/flu_hi_msa_regressor/before_$year-$month/"$subtype"_seed=1005/random_split/max_steps_150k/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)  # training longer?

echo $lm_ckpt
echo $hi_ckpt

year_minus_three=`(expr $year - 3)`
year_plus_one=`(expr $year + 1)` 

index=`expr \( $year - 2018 \) \* 2 + 30`
testing_time=$index

if [[ $lm_ckpt == *"_2M"* ]];
then
    min_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 5`
    max_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 7`
else
    min_testing_time=$testing_time
    max_testing_time=$testing_time
fi

# [1]
# Candidate vaccines = 3Y
candidate_vaccine_path="../data/gisaid/ha_processed/$year_minus_three-"$month"_to_$year-"$month"_9999M/$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.fasta"
# Testing viruses = 3Y
testing_viruses_path="../data/gisaid/ha_processed/$year_minus_three-$month""_to_$year-$month""_9999M/$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.fasta"
working_dir="../runs/pipeline/$year-$month/$subtype/vaccine_set=$year_minus_three-$month-$year-$month""___virus_set=$year_minus_three-$month-$year-$month"

bash pipeline/run.sh --candidate_vaccine_path $candidate_vaccine_path --testing_viruses_path $testing_viruses_path --working_directory $working_dir --devices $device, --hi_predictor_ckpt $hi_ckpt --min_testing_time $min_testing_time --max_testing_time $max_testing_time --domiance_predictor_ckpt $lm_ckpt

# [2]
# Vaccine candidate = CDC
candidate_vaccine_path="../data/recommended_vaccines_from_gisaid_ha/$year-$year_plus_one"_NH_"$subtype.fasta"
# Testing viruses = 3Y
testing_viruses_path="../data/gisaid/ha_processed/$year_minus_three-$month""_to_$year-$month""_9999M/$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.fasta"
working_dir="../runs/pipeline/$year-$month/$subtype/vaccine_set=who___virus_set=$year_minus_three-$month-$year-$month"

bash pipeline/run.sh --candidate_vaccine_path $candidate_vaccine_path --testing_viruses_path $testing_viruses_path --working_directory $working_dir --devices $device, --hi_predictor_ckpt $hi_ckpt --min_testing_time $min_testing_time --max_testing_time $max_testing_time --domiance_predictor_ckpt $lm_ckpt

# [3] optional -> used in drawing phylo trees
# index=`expr \( $year - 2018 \) \* 2 + 30`
# # Vaccine candidate = 3Y
# candidate_vaccine_path="../data/gisaid/ha_processed/$year_minus_three-"$month"_to_$year-"$month"_9999M/$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.fasta"
# # Testing viruses = ground-truth
# testing_viruses_path="../data/gisaid/ha_processed/2003-10_to_2023-04_6M/$subtype/human_minBinSize100_lenQuantile0.2_bins/$index.fasta"
# working_dir="../runs/pipeline/$year-$month/$subtype/vaccine_set=$year_minus_three-$month-$year-$month___virus_set=$index"

# bash pipeline/run.sh --candidate_vaccine_path $candidate_vaccine_path --testing_viruses_path $testing_viruses_path --working_directory $working_dir --devices $device, --hi_predictor_ckpt $hi_ckpt --min_testing_time $min_testing_time --max_testing_time $max_testing_time --domiance_predictor_ckpt $lm_ckpt
