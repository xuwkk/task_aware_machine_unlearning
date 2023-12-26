# Task-aware Machine Unlearning

**The paper is under review. More codes are coming.**

This repo contains data and code for Task-Aware Machine Unlearning with Application to Load Forecasting. The authors are from Control and Power Research Group, Department of EEE, Imperial College London.

You can find the paper by this [arxiv preprint](https://arxiv.org/abs/2308.14412).

**Abstract**:
Data privacy and security have become a non-negligible factor in load forecasting. Previous researches mainly focus on training stage enhancement. However, once the model is trained and deployed, it may need to `forget' (i.e., remove the impact of) part of training data if the data is found to be malicious or as requested by the data owner. This paper introduces machine unlearning algorithm which is specifically designed to remove the influence of part of the original dataset on an already trained forecaster. However, direct unlearning inevitably degrades the model generalization ability. To balance between unlearning completeness and performance degradation, a performance-aware algorithm is proposed by evaluating the sensitivity of local model parameter change using influence function and sample re-weighting. Moreover, we observe that the statistic criterion cannot fully reflect the operation cost of down-stream tasks. Therefore, a task-aware machine unlearning is proposed whose objective is a tri-level optimization with dispatch and redispatch problems considered. We theoretically prove the existence of the gradient of such objective, which is key to re-weighting the remaining samples. We test the unlearning algorithms on linear and neural network load forecasters with realistic load dataset. The simulation demonstrates the balance on unlearning completeness and operational cost.

Please cite our paper if it helps your research:
```
@article{xu2023task,
  title={Task-Aware Machine Unlearning and Its Application in Load Forecasting},
  author={Xu, Wangkun and Teng, Fei},
  journal={arXiv preprint arXiv:2308.14412},
  year={2023}
}
```

## Package

The key packages used in this repo include:

- `torch==2.0.1+cpu` for automatic differentiation.
- `cvxpy==1.2.3` for formulating convex optimization problem, refering to [here](https://www.cvxpy.org/).
- `cvxpylayer==0.1.5` for differentiating the convex layer, refering [here](https://github.com/cvxgrp/cvxpylayers).
- a modified [torch-influence](https://github.com/alstonlo/torch-influence) to calculate the influence function. Our implementation can be found at `torch_influence/'.
- `gurobipy==11.0.0` (optional) for solving optimization problems.

We note that the higher versions of `cvxpy` and `cvxpylayers` may not work. All the experiments are on cpu.


## Prepare

### Data source

We use open source dataset from [A Synthetic Texas Backbone Power System with Climate-Dependent Spatio-Temporal Correlated Profiles](https://arxiv.org/abs/2302.13231). You can download/read the descriptions of the dataset from [the official webpage](https://rpglab.github.io/resources/TX-123BT/). *Please cite/recogenize the original authors if you use the dataset.*

After download the the `.zip` file into the `data/` and change the name into `raw_data.zip`, unzip the file by 
```bash
unzip data/raw_data.zip -d data/
```

This will give a new folder `data/Data_public/`

### Configuration

The configurations of all power grid, network, and training are summarized in `conf/`. We use python package [hydra](https://hydra.cc/) to organize the experiments.

### Data preprocessing

To generate the data for IEEE bus-14 system, run this by default (this may take several minutes)
```bash
python clean_data.py
```

It will generate 14 `.cxv` files in `data/data_case14` folder. We run a simple algorithm to find the most uncorrelated 14 loads from the original dataset. The algorithm is implemented in `clean_data.py`. Each `.csv` file contains the features and target load for one bus. Note that the calendric features are represented by the sine and cosine functions with different periods. The meteorological features are normalized by the mean and the variance. More detailed preprocessing can be found in the paper.

In addition, the raw load data in the Texas Backbone Power System cannot be directly used for arbitrary IEEE test system. Therefore, we first preprocess the load data to match the load level of system under test.

We have made data preprocessing flexible. You can also modify the `clean_data.py` and the `case_modifier()` function in `utils/dataaset/` to match your system. In the future, we will update the data generation procedure to support more power grid.

### Train the load forecaster

The linear load forecaster can be automatically trained when any unlearning algorithms are called. However, you need to train the neural network baseds forecaster on the core data. We provide two NN structure, one is naive convolutional nn and another is [MLP-Mixer](https://arxiv.org/abs/2105.01601).

To train the NN forecaster on the **core dataset**, you can run
```bash
python train_nn.py model=conv
```
and
```bash
python train_nn.py model=mlpmixer
```

## Unlearning

### Check the results

Our code contains a simple check for unlearning the `linear` models using
1. Analytic method which calculates the gradient and Hessian analytically
2. Call `torch-influence` functions to calculate the gradient, Hessian, and inverse Hessian vector product (iHVP) using `direct` and `cg`.
3. Different unlearning formulations based on our paper.

To check the all the implementations are correct, run
```bash
python check.py unlearn_prop={a float number} model={linear or conv or mlpmixer}
```

### Influence function based unlearning

To highlight the significance of the unlearning performance, we choose the samples that can significantly improve the expected test set performance to unlearn.

To find the impact of each training sample on the expected test performance, run
```bash
python gen_index.py model={linear, conv, mlpmixer}
```

To run the influence function based unlearning (as baseline)
```bash
python eval_unlearn.py model={linear, conv, mlpmixer}, unlearn_prop={a float number}
```

### Performance aware and task aware machine unlearning (PUMU and TAMU)
Set the criteria property to unchange the performance of mse, mape, or cost.
```
python eval_unchange.py unlearn_prop={a float number} model={linear, conv, mlpmixer} criteria={mse, mape, cost}
```