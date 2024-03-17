
getLMcheckpoints(){
  subtype=$1
  year=$2

  case $subtype in
    "h1n1" )
    case $year in
        "2011" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2011-04_6M/a_h1n1/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=87-step=14872.ckpt" ;;
        "2012" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2012-04_6M/a_h1n1/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=97-step=18620.ckpt" ;;
        "2013" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2013-04_6M/a_h1n1/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=90-step=20111.ckpt" ;;
        "2014" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2014-04_6M/a_h1n1/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=82-step=21414.ckpt" ;;
        "2015" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2015-04_6M/a_h1n1/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=97-step=27440.ckpt" ;;
        "2016" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2016-04_6M/a_h1n1/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=98-step=34056.ckpt" ;;
        "2017" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2017-04_6M/a_h1n1/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=97-step=36652.ckpt" ;;
        "2018" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2018-04_6M/a_h1n1/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=89-step=39240.ckpt" ;;
        "2019" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2019-04_6M/a_h1n1/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=93-step=51794.ckpt" ;;
        "2020" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2020-04_6M/a_h1n1/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=73-step=48100.ckpt" ;;
        "2021" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2021-04_6M/a_h1n1/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=68-step=44988.ckpt" ;;
        "2022" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2022-04_6M/a_h1n1/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=93-step=62510.ckpt" ;;
        esac
    ;;

    "h3n2" )
    case $year in
        "2011" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2011-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=97-step=6958.ckpt" ;;
        "2012" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2012-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=91-step=9384.ckpt" ;;
        "2013" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2013-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=97-step=13524.ckpt" ;;
        "2014" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2014-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=98-step=16236.ckpt" ;;
        "2015" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2015-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=95-step=21888.ckpt" ;;
        "2016" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2016-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=94-step=26505.ckpt" ;;
        "2017" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2017-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=95-step=37248.ckpt" ;;
        "2018" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2018-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=98-step=48312.ckpt" ;;
        "2019" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2019-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=99-step=58400.ckpt" ;;
        "2020" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2020-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=93-step=62980.ckpt" ;;
        "2021" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2021-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=92-step=62868.ckpt" ;;
        "2022" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2022-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=82-step=61669.ckpt" ;;
        esac
    ;;

    esac
    echo "$lm_ckpt"
}

getHIcheckpoints(){
  subtype=$1
  year=$2

  case $subtype in
    "h1n1" )
    case $year in
        "2011" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2011-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_0/checkpoints/epoch=93-step=20774.ckpt" ;;
        "2012" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2012-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_1/checkpoints/epoch=68-step=24771.ckpt" ;;
        "2013" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2013-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=99-step=58500.ckpt" ;;
        "2014" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2014-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=85-step=72670.ckpt" ;;
        "2015" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2015-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=81-step=90528.ckpt" ;;
        "2016" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2016-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=93-step=130284.ckpt" ;;
        "2017" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2017-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=83-step=132720.ckpt" ;;
        "2018" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2018-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=94-step=180975.ckpt" ;;
        "2019" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2019-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=84-step=201790.ckpt" ;;
        "2020" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2020-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=88-step=245907.ckpt" ;;
        "2021" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2021-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=97-step=273322.ckpt" ;;
        "2022" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2022-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=94-step=289085.ckpt" ;;
        esac
    ;;

    "h3n2" )
    case $year in
        "2011" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2011-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_0/checkpoints/epoch=93-step=20774.ckpt" ;;
        "2012" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2012-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_1/checkpoints/epoch=68-step=24771.ckpt" ;;
        "2013" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2013-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=99-step=58500.ckpt" ;;
        "2014" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2014-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=85-step=72670.ckpt" ;;
        "2015" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2015-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=81-step=90528.ckpt" ;;
        "2016" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2016-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=93-step=130284.ckpt" ;;
        "2017" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2017-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=83-step=132720.ckpt" ;;
        "2018" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2018-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=94-step=180975.ckpt" ;;
        "2019" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2019-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=84-step=201790.ckpt" ;;
        "2020" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2020-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=88-step=245907.ckpt" ;;
        "2021" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2021-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=97-step=273322.ckpt" ;;
        "2022" )
        lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2022-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_2/checkpoints/epoch=94-step=289085.ckpt" ;;
        esac
    ;;

    esac
    echo "$lm_ckpt"
}


subtype=$1
year=$2
device=$3

lm_ckpt=$(getLMcheckpoints $subtype $year)
hi_ckpt=$(getHIcheckpoints $subtype $year)
echo $lm_ckpt
echo $hi_ckpt

# year="2018"
year_minus_three=`(expr $year - 3)` # TODO: 
year_plus_one=`(expr $year + 1)` # TODO: 
# subtype="h1n1"
index=`expr \( $year - 2018 \) \* 2 + 30`

candidate_vaccine_path="/data/rsg/nlp/wenxian/esm/data/gisaid/flu/cdc_vaccine_strains/cdc_recommend_vaccines_from_gisaid_ha/$year-$year_plus_one"_NH_"a_$subtype.fasta"
# candidate_vaccine_path="/data/rsg/nlp/wenxian/esm/data/gisaid/flu/ha_processed/$year_minus_three-04_to_$year-04_9999M/a_$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.fasta"
gt_hi_path="/data/rsg/nlp/wenxian/esm/data/who_flu/a_"$subtype"_hi_folds.csv"
gt_dominamce_path="/data/rsg/nlp/wenxian/esm/data/gisaid/flu/ha_processed/2003-10_to_2023-04_6M/a_$subtype/human_minBinSize100_lenQuantile0.2_bins/$index.fasta"
testing_time=$index

# Testing viruses = 3Y

testing_viruses_path="/data/rsg/nlp/wenxian/esm/data/gisaid/flu/ha_processed/$year_minus_three-04_to_$year-04_9999M/a_$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.fasta"
working_dir="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/pipeline/$year-04/a_$subtype/vaccine_set=who___virus_set=$year_minus_three-04-$year-04"

bash pipeline/run.sh --candidate_vaccine_path $candidate_vaccine_path --testing_viruses_path $testing_viruses_path --working_directory $working_dir --devices $device, --hi_predictor_ckpt $hi_ckpt --testing_time $testing_time --domiance_predictor_ckpt $lm_ckpt --gt_hi_path $gt_hi_path --gt_dominamce_path $gt_dominamce_path



# 2018
# hi_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2018-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_1/checkpoints/epoch=87-step=144232.ckpt"
# lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2018-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2_tune/lr=1e-4/lightning_logs/version_0/checkpoints/epoch=68-step=28014.ckpt" # H3N2
# lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2018-04_6M/a_h1n1/human_minBinSize1000_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=84-step=27965.ckpt" # H1N1

# # 2019:
# hi_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2019-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_0/checkpoints/epoch=92-step=192882.ckpt"
# lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2019-04_6M/a_h3n2/human_minBinSize1000_lenQuantile0.2/lightning_logs/version_1/checkpoints/epoch=95-step=48192.ckpt"

# # 2020:
# hi_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2020-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_0/checkpoints/epoch=81-step=201966.ckpt"
# lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2020-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=93-step=62980.ckpt"

# # 2021:
# hi_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2021-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_0/checkpoints/epoch=85-step=214054.ckpt"
# lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2021-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=92-step=62868.ckpt"
# # 2022:
# # hi_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_hi_msa_regressor/before_2022-04/a_h1n1_and_h3n2_seed=0/random_split/lightning_logs/version_0/checkpoints/epoch=79-step=219440.ckpt"
# lm_ckpt="/data/rsg/nlp/wenxian/esm/devo_lightning/runs/flu_lm/2003-10_to_2022-04_6M/a_h3n2/human_minBinSize100_lenQuantile0.2/lightning_logs/version_0/checkpoints/epoch=82-step=61669.ckpt"
