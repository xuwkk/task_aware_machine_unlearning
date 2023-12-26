"""
evaluate the direct unlearning performance on
1. mse
2. mape
3. cost
of the following models
1. linear model
2. nn_conv model on the last linear layer
3. nn_mlpmixer model on the last linear layer

For the linear model, this file can automatically train a linear regressor for load forecasting.
For the nn model, you need to train a neural network first before running this file.

To highlight the significance, three options can be used for generating the unlearning set
construct the unlearning set based on the choice of {random, helpful, harmful}
1. helpful: if removed, the test set mape will be significantly increased (performance gets worse).
2. harmful: if removed, the test set mape will be significantly decreased (performance gest worse).
3. random: randomly select the unlearning set (less significant).
"""

import json
import argparse
from torch.utils.data import DataLoader
from utils import return_dataset, evaluate, reconstruct_model, flatten_model
from func_operation import return_unlearn_datasets, return_trained_model, return_module, return_core_datasets, return_dataset_for_nn_affine
import numpy as np
import torch
import os
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    unlearn_prop = cfg.unlearn_prop
    batch_size = cfg.data.batch_size_eval
    model_type = cfg.model.type
    influence_dir = cfg.model.influence_dir
    save_dir = cfg.simulation_dir + f"{model_type}/"
    train_loss = cfg.model.train_loss

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # dataset
    influences = torch.from_numpy(np.load(influence_dir)).float()
    dataset_train, dataset_test = return_dataset(cfg)
    if 'nn' in model_type:
        dataset_core, dataset_sensitive = return_core_datasets(cfg, dataset_to_be_split=dataset_train)
        dataset_train, dataset_test = return_dataset_for_nn_affine(cfg, dataset_sensitive, dataset_test)
    
    dataset_unlearn, dataset_remain, _, _ = return_unlearn_datasets(influences=influences, 
                                                            unlearn_prop=unlearn_prop, dataset_to_be_unlearn=dataset_train, 
                                                            mode=cfg.unlearn_mode,
                                                            config=cfg)
    loader_remain = DataLoader(dataset_remain, batch_size = batch_size, shuffle = False)  # only unlearn using the remain set formulation

    print('Load is scaled:\n', dataset_train.is_scale, dataset_test.is_scale, dataset_unlearn.is_scale, dataset_remain.is_scale)
    print('train dataset shape: ', dataset_train.feature.shape, 'test dataset shape: ', dataset_test.feature.shape)

    # model
    model_ori = return_trained_model(cfg,
                                    model_type = model_type + "-affine" if 'nn' in model_type else model_type,
                                    dataset_train = dataset_train, is_spo = False)
    model_retrain = return_trained_model(cfg,
                                    model_type = model_type + "-affine" if 'nn' in model_type else model_type, 
                                    dataset_train = dataset_remain, is_spo = False)

    model_ori.eval()
    model_retrain.eval()
    parameter_ori = flatten_model(model_ori)
    parameter_retrain = flatten_model(model_retrain)

    print('number of parameters: ', parameter_ori.shape, parameter_retrain.shape)


    # machine unlearning
    module = return_module(
        cfg,
        loss_type_dict = {"train":train_loss, "test": train_loss},
        loader_dict={"train": loader_remain, "test": loader_remain}, 
        model=model_ori, method = 'cg', watch_progress=False
    )

    ihvp = module.stest(test_idxs=range(len(dataset_remain))).numpy()
    parameter_unlearn = parameter_ori - ihvp
    model_unlearn = reconstruct_model(model = model_ori, flattened_params = parameter_unlearn)

    print('max difference between unlearn and retrain: ', np.max(np.abs(parameter_retrain - parameter_unlearn)))

    dataset_collection = {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}

    mse_ori = evaluate(model_ori, dataset_collection, loss = 'mse')
    mape_ori = evaluate(model_ori, dataset_collection, loss = 'mape')
    cost_ori = evaluate(model_ori, dataset_collection, loss = 'cost', case_config = cfg.case)

    print('original model:')
    print("mse:\n", mse_ori, "\nmape:\n", mape_ori, "\ncost:\n", cost_ori)

    mse_unlearn = evaluate(model_unlearn, dataset_collection, loss = 'mse')
    mape_unlearn = evaluate(model_unlearn, dataset_collection, loss = 'mape')
    cost_unlearn = evaluate(model_unlearn, dataset_collection, loss = 'cost', case_config = cfg.case)

    print('unlearned model:')
    print("mse:\n", mse_unlearn, "\nmape:\n", mape_unlearn, "\ncost:\n", cost_unlearn)

    mse_retrain = evaluate(model_retrain, dataset_collection, loss = 'mse')
    mape_retrain = evaluate(model_retrain, dataset_collection, loss = 'mape')
    cost_retrain = evaluate(model_retrain, dataset_collection, loss = 'cost', case_config = cfg.case)

    print('retrained model:')
    print("mse:\n", mse_retrain, "\nmape:\n", mape_retrain, "\ncost:\n", cost_retrain)

    metric = {}
    metric["mse_ori"] = mse_ori
    metric["mape_ori"] = mape_ori
    metric["cost_ori"] = cost_ori
    metric["mse_unlearn"] = mse_unlearn
    metric["mape_unlearn"] = mape_unlearn
    metric["cost_unlearn"] = cost_unlearn
    metric["mse_retrain"] = mse_retrain
    metric["mape_retrain"] = mape_retrain
    metric["cost_retrain"] = cost_retrain

    np.save(save_dir + f"unlearning_{unlearn_prop}.npy", metric)

if __name__ == "__main__":

    main()
