# VaxSeer: ML-Optimized Influenza Vaccine Selection

## Overview

VaxSeer is a machine learning system designed to optimize seasonal influenza vaccine strain selection. The project aims to improve vaccine effectiveness by predicting which influenza virus strains should be included in seasonal flu vaccines to maximize protection against future circulating viruses.

### Key Features

- **Dual Prediction System**: 
  - **Dominance Predictor**: Forecasts which flu strains will become dominant in upcoming seasons
  - **Antigenicity Predictor**: Measures cross-protection between vaccine strains and circulating viruses using HI (hemagglutination inhibition) test data

- **Coverage Score Optimization**: Calculates vaccine coverage scores to predict population-level protection

- **Multi-subtype Support**: Handles both H3N2 and H1N1 influenza subtypes

- **Data-Driven Approach**: Trained on historical data (2012-2021) using GISAID sequence databases

- **Retrospective Evaluation**: Validates predictions against actual WHO vaccine recommendations and real-world vaccine effectiveness

The system processes viral hemagglutinin (HA) protein sequences to predict future strain dominance and antigenic relationships, potentially reducing disease burden from seasonal influenza through improved vaccine strain selection.

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
```
Results are available [here](https://people.csail.mit.edu/wxsh/vaxseer/results.tar.gz).
```
wget https://people.csail.mit.edu/wxsh/vaxseer/results.tar.gz

tar -zxvf results.tar.gz
```