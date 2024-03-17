

# Training dominance predictor

`run_scripts/train_domiance_predictor.sh` gives an example of training dominance predictor. To train the dominance predictor for 2018, you could try

```
year="2018"
subtype="a_h3n2"
gpu="0"
ckpt_dir="../runs/flu_lm"

bash run_scripts/train_domiance_predictor.sh $year $subtype $gpu $ckpt_dir
```

# Training antigenicity predictor

`run_scripts/train_hi_predictor.sh` gives an example of training antigenicity (HI test) predictor. To train the antigenicity predictor for 2018, you could try

```
year="2018"
subtype="a_h3n2"
gpu="0"
ckpt_dir="../runs/flu_hi_msa_regressor"

bash run_scripts/train_hi_predictor.sh $year $subtype $gpu $ckpt_dir
```