
year=$1 # "2018"
month=$2 # 02

raw_pairs_dir="hi_processed"
sequences_path="../gisaid/ha.fasta"
seed="1005"
all_subtypes=("a_h3n2" "a_h1n1")

max_date_exclude="$year-$month"

for subtype in "${all_subtypes[@]}"
do
    echo "Processing $subtype"
    raw_pairs_path="$raw_pairs_dir/"$subtype"_pairs.csv"
    
    if [ ! -f "$raw_pairs_dir/before_$max_date_exclude"/"$subtype"_hi_folds.csv ]; then
        python prepare.py --pairs_path $raw_pairs_path --max_date_exclude $max_date_exclude --sequences_path $sequences_path --output_root_dir $raw_pairs_dir
    fi

    if [ ! -f "$raw_pairs_dir/before_$max_date_exclude"/"$subtype".m8 ]; then
        mmseqs easy-search $raw_pairs_dir/before_$max_date_exclude/"$subtype"_vaccine.fasta $raw_pairs_dir/before_$max_date_exclude/"$subtype"_virus.fasta $raw_pairs_dir/before_$max_date_exclude/$subtype.m8 mmseqs_tmp --format-output "query,target,qaln,taln,qstart,qend,tstart,tend,mismatch" --max-seqs 5000

        rm -r mmseqs_tmp
    fi

    if [ ! -d "$raw_pairs_dir/before_$max_date_exclude"/$subtype"_seed=$seed" ]; then
        echo "splitting real values..."
        python split.py --pairs_dir "$raw_pairs_dir/before_$max_date_exclude" --subtype $subtype --seed $seed
    fi

done