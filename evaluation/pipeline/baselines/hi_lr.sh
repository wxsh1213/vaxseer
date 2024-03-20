month="02"
split="random_split"

ckpt_saving_root_dir="../runs"

for year in `seq 2012 2014`
do
    year_minus_three=`(expr $year - 3)`
    for subtype in "a_h3n2" "a_h1n1"
    do
        echo ">>>>>" $split $subtype $year
        
        working_directory="../runs/pipeline/"$year"-"$month"/$subtype/vaccine_set=`(expr $year - 3)`-"$month"-$year-"$month"___virus_set=`(expr $year - 3)`-"$month"-$year-"$month""
        test_path="$working_directory/vaccine_virus_pairs/pairs.csv"

        training_path="../data/antigenicity/hi_processed/before_$year-"$month"/$subtype""_seed=1005/$split/train.csv"
        valid_path="../data/antigenicity/hi_processed/before_$year-"$month"/$subtype""_seed=1005/$split/valid.csv"
        
        ckpt_saving_dir="$ckpt_saving_root_dir/flu_hi_linear_regressor/before_$year-"$month"/$subtype""_seed=1005/$split"
        alignment_path="../data/antigenicity/hi_processed/before_$year-"$month"/$subtype.m8"
        vaccine_ref_aln_path="../data/antigenicity/hi_processed/before_$year-"$month"/$subtype""_vaccine.ref.m8"
        virus_ref_aln_path="../data/antigenicity/hi_processed/before_$year-"$month"/$subtype""_virus.ref.m8"
        
        testing_viruses_path="../data/gisaid/ha_processed/$year_minus_three-"$month"_to_$year-"$month"_9999M/$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.fasta"
        testing_viruses_ref_aln_path="../data/gisaid/ha_processed/$year_minus_three-"$month"_to_$year-"$month"_9999M/$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.ref.m8" 
        
        tesing_candidate_vaccine_path="../data/gisaid/ha_processed/$year_minus_three-"$month"_to_$year-"$month"_9999M/$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.fasta"
        testing_candidate_vaccine_ref_aln_path="../data/gisaid/ha_processed/$year_minus_three-"$month"_to_$year-"$month"_9999M/$subtype/human_minBinSize1000_lenQuantile0.2_minCnt5.ref.m8"

        if [[ $subtype == "a_h3n2" ]]
        then 
            reference_fasta_path="../data/reference_fasta/prortein_A_NewYork_392_2004_H3N2_ha.fasta"
        fi

        if [[ $subtype == "a_h1n1" ]]
        then 
            reference_fasta_path="../data/reference_fasta/protein_A_California_07_2009_H1N1_ha.fasta"
        fi


        if [ ! -f $vaccine_ref_aln_path ]; 
        then 
            vaccine_fasta_path="../data/antigenicity/hi_processed/before_$year-"$month"/$subtype""_vaccine.fasta"
            mmseqs easy-search $vaccine_fasta_path $reference_fasta_path $vaccine_ref_aln_path tmp --format-output "query,target,qaln,taln,qstart,qend,tstart,tend,mismatch" --max-seqs 2000
        fi

        if [ ! -f $virus_ref_aln_path ]; 
        then 
            virus_fasta_path="../data/antigenicity/hi_processed/before_$year-"$month"/$subtype""_virus.fasta"
            mmseqs easy-search $virus_fasta_path $reference_fasta_path $virus_ref_aln_path tmp --format-output "query,target,qaln,taln,qstart,qend,tstart,tend,mismatch" --max-seqs 2000
        fi

        # Align testing viruses sequences to the reference sequence
        if [ ! -f $testing_viruses_ref_aln_path ];
        then
            mmseqs easy-search $testing_viruses_path $reference_fasta_path $testing_viruses_ref_aln_path tmp --format-output "query,target,qaln,taln,qstart,qend,tstart,tend,mismatch" --max-seqs 2000
        fi

        # Align vaccine sequences to the reference sequence
        if [ ! -f $testing_viruses_ref_aln_path ];
        then
            mmseqs easy-search $tesing_candidate_vaccine_path $testing_candidate_vaccine_ref_aln_path $testing_viruses_ref_aln_path tmp --format-output "query,target,qaln,taln,qstart,qend,tstart,tend,mismatch" --max-seqs 2000
        fi

        testing_result_saving_path="$working_directory/vaccine_virus_pairs/prediction/aa_subs/predictions.csv"

        # cd ../baselines
        # --alignment_path $alignment_path
        python ./baselines/hi_linear_regression.py --model "neher" --training_path $training_path --valid_path $valid_path --test_path $test_path --ckpt_saving_dir $ckpt_saving_dir  --train_vaccine_ref_aln_path $vaccine_ref_aln_path --train_virus_ref_aln_path $virus_ref_aln_path --valid_vaccine_ref_aln_path $vaccine_ref_aln_path --valid_virus_ref_aln_path $virus_ref_aln_path --test_vaccine_ref_aln_path $testing_candidate_vaccine_ref_aln_path --test_virus_ref_aln_path $testing_viruses_ref_aln_path --testing_result_saving_path $testing_result_saving_path --ref_seq_path $reference_fasta_path

    done
done