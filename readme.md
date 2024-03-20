# Setup

### Environment

```
conda env create -f environment.yaml
conda activate vaxseer
```

[MMseq2](https://github.com/soedinglab/MMseqs2) is required for aligning sequences.


# Data

In [data/readme.md](data/readme.md).

# Model training

In [vaxseer/readme.md](vaxseer/readme.md).

# Retrospective evaluation

In [evaluation/readme.md](evaluation/readme.md). 

# Models and results

Models are available [here](https://www.dropbox.com/scl/fo/7d94eqsii2h1jdm5l7mm6/h?rlkey=1n1wafyuapwx5a4c04jc0y7cs&dl=0).

One could also run the following code to download specific models:

```
# task: lm (dominance predictor) or/and hi (antigenicity predictor)
# year: from 2012 to 2021
# subtype: a_h3n2 or/and a_h1n1

python download_models_from_dropbox.py --task lm --year 2012 --subtype a_h3n2 --output_dir runs

# To download all models (around 60GB):

python download_models_from_dropbox.py 

# extract files

cd runs/flu_lm
tar -zxvf *.tar.gz

cd ../flu_hi_msa_regressor
tar -zxvf *.tar.gz
```

Results are available [here](https://people.csail.mit.edu/wxsh/vaxseer/results.tar.gz).

```
wget https://people.csail.mit.edu/wxsh/vaxseer/results.tar.gz

tar -zxvf results.tar.gz
```