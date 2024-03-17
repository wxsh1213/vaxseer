month="02"

start_year="2015"
end_year="2021"

ckpt_saving_root_dir="../runs"

for year in `seq $start_year $end_year`; 
do
    for subtype in "a_h3n2" "a_h1n1"
    do
        echo ">>>>>" $year $subtype

        index=`expr \( $year - 2018 \) \* 2 + 30`

        if [[ $subtype == "a_h3n2" ]]
        then 
            reference_fasta_path="../data/reference_fasta/prortein_A_NewYork_392_2004_H3N2_ha.fasta"
        fi
        if [[ $subtype == "a_h1n1" ]]
        then 
            reference_fasta_path="../data/reference_fasta/protein_A_California_07_2009_H1N1_ha.fasta"
        fi

        train_fasta_path="../data/gisaid/ha_processed/2003-10_to_$year-"$month"_2M/$subtype/human_minBinSize100_lenQuantile0.2.fasta"
        train_aln_path="../data/gisaid/ha_processed/2003-10_to_$year-"$month"_2M/$subtype/human_minBinSize100_lenQuantile0.2.ref.m8"

        if [ ! -f $train_aln_path ]; 
        then 
            mmseqs easy-search $train_fasta_path $reference_fasta_path $train_aln_path tmp --format-output "query,target,qaln,taln,qstart,qend,tstart,tend,mismatch" --max-seqs 2000
        fi

        ckpt_saving_dir="$ckpt_saving_root_dir/flu_lm_aa_subs/2003-10_to_$year-"$month"_2M/$subtype/human_minBinSize100_lenQuantile0.2"

        cd ../evaluation
        python baselines/dominance_aa_subs.py --train_fasta_path $train_fasta_path --ckpt_saving_dir $ckpt_saving_dir
    
    done
done