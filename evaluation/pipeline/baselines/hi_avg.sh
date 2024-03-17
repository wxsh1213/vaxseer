# year=$1 # "2018"
# subtype=$2 # "h1n1"
month="02" # "02"


for year in `seq 2012 2014`
do
    for subtype in "a_h1n1" "a_h3n2"
    do
        echo $year $subtype
        working_directory_list=("../runs/pipeline/"$year"-$month/$subtype/vaccine_set=who___virus_set=`(expr $year - 3)`-"$month"-$year-"$month""  "../runs/pipeline/"$year"-$month/$subtype/vaccine_set=`(expr $year - 3)`-"$month"-$year-"$month"___virus_set=`(expr $year - 3)`-"$month"-$year-"$month"")

        for working_directory in "${working_directory_list[@]}"
        do
            echo "working_directory:" $working_directory

            # Impute by the whole average
            exp_hi_save_dir="$working_directory/vaccine_virus_pairs/prediction/exp_before_"$year-$month"_avg"
            hi_exp_results="../data/antigenicity/hi_processed/before_"$year"-$month/"$subtype"_hi_folds.csv"
            pairs_save_path="$working_directory/vaccine_virus_pairs/pairs.csv"

            mkdir -p $exp_hi_save_dir
            python baselines/hi_from_ground_truth.py --hi_form_path $hi_exp_results --index_pair $pairs_save_path --save_path $exp_hi_save_dir"/predictions.csv"

        done
    done
done

