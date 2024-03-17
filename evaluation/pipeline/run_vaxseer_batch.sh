subtype="a_h3n2"
gpu_array=("0" "1" "2" "3" "4" "5")

for year in `seq 2013 2013`
do
    gpu=${gpu_array[$year - 2013]}
    echo $year $gpu
    nohup bash pipeline/run_vaxseer.sh $subtype $year $gpu > nohup.eval_vaxseer.$subtype.$year.log 2>&1 &
done