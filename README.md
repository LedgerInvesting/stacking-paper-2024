# BayesBlend real-world examples

This repository contains the scripts used to reproduce the analyses in "BayesBlend: Easy Model Blending 
using Pseudo-Bayesian Model Averaging, Stacking and Hierarchical Stacking in Python" by Haines & 
Goold (2024). 

## Install requirements

All analyses were run with `Python 3.11`. Once `Python 3.11` is installed locally, we recommend the 
following steps: 

1. navigate to your local `stacking-paper-2024/` directory
2. initialize a virtual environment: `python3.11 venv env`
3. activate the environment: `source env/bin/activate`
4. install requirements: `pip install -r requirements/requirements.txt`

## Reproduce analyses

Analyses can then be reproduced by running the following scripts in order: 

1. download and pre-process the data: `python -m data-prep`
2. fit the loss development models and produce figures: `python -m development`
3. fit the loss forecasting models and produce figures: `python -m forecast`

Once analyses are reproduced, figures are located in the `stacking-paper-2024/figures/` directory.
