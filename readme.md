# Task-aware Machine Unlearning

This repo contains data and code for Task-Aware Machine Unlearning with Application to Load Forecasting.

## Data

### Data source

We use open source dataset from [A Synthetic Texas Backbone Power System with Climate-Dependent Spatio-Temporal Correlated Profiles](https://arxiv.org/abs/2302.13231). You can download/read the descriptions of the dataset from [the official webpage](https://rpglab.github.io/resources/TX-123BT/). 

*Please cite/recogenize the original authors if you use the dataset.*

After download the the `.zip` file into the `data/` and change the name into `raw_data.zip`, unzip the file by 
```bash
unzip data/raw_data.zip -d data/
```

### Data preprocessing

To generate the data for IEEE bus-14 system, run
```bash
python clean_data.py --no_bus 14
```

It will generate 14 `.cxv` files in `data/data_case14` folder. We run a simple algorithm to find the most uncorrelated 14 loads from the original dataset. The algorithm is implemented in `clean_data.py`. Each `.csv` file contains the features and target load for one bus. Note that the calendric features are represented by the sine and cosine functions with different periods. The meteorological features are normalized by the mean and the variance.


## Usage

### Check the results

Our code contains a check for the unlearning on `affine` and `reverse-affine` models using
1. Analytic method which calculates the gradient and Hessian analytically
2. Call `torch-influence` functions to calculate the gradient, Hessian, and inverse Hessian vector product (iHVP) using `direct` and `cg`.

To check the results, run
```bash
# python check.py -u unlearn_percentage --model_type affine or reverse-affine
python check.py -u 0.2 --model_type affine
python check.py -u 0.2 --model_type nn
```

### Influence function based unlearning

To run the influence function based unlearning (as baseline)
```bash
# python eval_unlearn.py -u unlearn_percentage -m random or helpful or harmful --model affine or reverse-affine or nn
python eval_unlearn.py -u 0.2 --model affine
python eval_unlearn.py -u 0.2 --model nn
```

### Task-aware unlearning
```bash
python eval_unchange.py -u 0.25 --model affine -c mse
python eval_unchange.py -u 0.25 --model affine -c mape
python eval_unchange.py -u 0.25 --model affine -c cost
```
