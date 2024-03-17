
for year in `seq 2012 2021`
do
    bash antigenicity_pipeline.sh $year 02
done

# ALL
bash antigenicity_pipeline.sh 2023 04