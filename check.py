"""
check the unlearnt model with the trained model (there parameters should be the same) for:
1. linear model
2. linear model upon the trained neutal network
"""

import json
import argparse

# from utils.dataset import return_dataset
# from utils.funcs import flatten_model
from func_operation import return_unlearn_datasets, return_trained_model, return_module, return_core_datasets, return_dataset_for_nn_affine
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from utils import return_dataset, flatten_model
from torch.utils.data import DataLoader

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    cfg_model = cfg.model
    cfg_data = cfg.data
    cfg_case = cfg.case

    print('check the unlearning performance with proportion: ', cfg.unlearn_prop)

    model_type = cfg_model.type
    batch_size = cfg_data.batch_size_eval

    """
    generate the dataset: unlearning random subset of the training dataset
    """
    unlearn_prop = cfg.unlearn_prop
    dataset_train, dataset_test = return_dataset(configs=cfg)

    if 'nn' in model_type:
        # do core, sensitive dataset split
        dataset_core, dataset_sensitive = return_core_datasets(cfg, dataset_to_be_split=dataset_train)
        # train and unlearn on the sensitive dataset
        dataset_train, dataset_test = return_dataset_for_nn_affine(cfg, dataset_sensitive, dataset_test)

    # random generate the unlearn dataset
    dataset_unlearn, dataset_remain, _, _ = return_unlearn_datasets(
                                            influences=None, 
                                            unlearn_prop=unlearn_prop, dataset_to_be_unlearn=dataset_train, 
                                            mode="random",
                                            config = cfg
                                            ) 

    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = False)
    loader_unlearn = DataLoader(dataset_unlearn, batch_size = batch_size, shuffle = False)
    loader_remain = DataLoader(dataset_remain, batch_size = batch_size, shuffle = False)

    print("shape:")
    print("train: ", dataset_train.feature.shape, "test: ", dataset_test.feature.shape, 
        "unlearn: ", dataset_unlearn.feature.shape, "remain: ", dataset_remain.feature.shape)

    """
    construct the model
    """
    model_ori = return_trained_model(cfg, model_type = model_type + "-affine" if 'nn' in model_type else model_type,
                                    dataset_train = dataset_train, is_spo = False)
    model_retrain = return_trained_model(cfg, model_type = model_type + "-affine" if 'nn' in model_type else model_type, 
                                        dataset_train = dataset_remain, is_spo = False)

    parameter_ori = flatten_model(model_ori)
    parameter_retrain = flatten_model(model_retrain)

    print('shape of the parameter:')
    print('original:', parameter_ori.shape, 'retrain:', parameter_retrain.shape)

    """
    test the gradient on the train dataset (which should be close to zero)
    """
    module = return_module(
                        configs=cfg,
                        loss_type_dict={'train':'mse', 'test':'mse'}, 
                        loader_dict = {'train': loader_train, 'test': loader_train},
                        model = model_ori, 
                        method = 'cg')

    train_grad = module.test_loss_grad(test_idxs = range(len(dataset_train))) # test loss test loader

    print("the shape of the train gradient is: ", train_grad.shape)
    print("the max train gradient is: ", torch.max(torch.abs(train_grad)).item())

    # assert torch.allclose(train_grad, torch.zeros_like(train_grad), atol=1e-5), "train gradient should be zero"
    print("===============================================")


    """
    unlearning result comparison
        hessian: remain
        gradient: remain
    """
    print("Unlearning performance. Hessian: remain, grad: remain.")
    module = return_module(configs=cfg,
                        loss_type_dict={'train':'mse', 'test':'mse'},
                        loader_dict = {'train': loader_remain, 'test': loader_remain},
                        model = model_ori, method = 'cg', watch_progress = False)

    ihvp = module.stest(test_idxs = range(len(dataset_remain)))
    parameter_unlearn = parameter_ori - ihvp.numpy()

    print('the max difference between parameters: ', np.max(np.abs(parameter_retrain - parameter_unlearn)))
    assert np.isclose(parameter_unlearn, parameter_retrain, atol=1e-4).all(), "parameter should be the same"
    print("===============================================")

    print("Test direct hvp")
    module = return_module(configs=cfg,
                        loss_type_dict={'train':'mse', 'test':'mse'},
                        loader_dict = {'train': loader_remain, 'test': loader_remain},
                        model = model_ori, method = 'direct')

    ihvp = module.stest(test_idxs = range(len(dataset_remain)))
    parameter_unlearn = parameter_ori - ihvp.numpy()
    print(  'the max difference between parameters: ', np.max(np.abs(parameter_retrain - parameter_unlearn)))
    assert np.isclose(parameter_unlearn, parameter_retrain, atol=1e-4).all(), "parameter should be the same"
    print("===============================================")

    """
    unlearning result comparison
        hessian: remain
        gradient: unlearn
    """
    print("Unlearning performance. Hessian: remain, grad: unlearn.")
    module = return_module(configs=cfg,
                        loss_type_dict={'train':'mse', 'test':'mse'},
                        loader_dict = {'train': loader_remain, 'test': loader_unlearn},
                        model = model_ori, method = 'cg')

    ihvp = module.stest(test_idxs = range(len(dataset_unlearn))) * len(dataset_unlearn) / len(dataset_remain) # ! dont forget to scale
    parameter_unlearn = parameter_ori + ihvp.numpy()  # ! + not -
    print(  'the max difference between parameters: ', np.max(np.abs(parameter_retrain - parameter_unlearn)))
    assert np.isclose(parameter_unlearn, parameter_retrain, atol=1e-4).all(), "parameter should be the same"

    print("===============================================")
    """
    unlearning result comparison
        hessian: train
        gradient: remain
    """
    print("Unlearning performance. Hessian: train, grad: remain.")
    module = return_module(configs=cfg,
                        loss_type_dict={'train':'mse', 'test':'mse'},
                        loader_dict = {'train': loader_train, 'test': loader_remain},
                        model = model_ori, method = 'cg')
    ihvp = module.stest(test_idxs = range(len(dataset_remain)))
    parameter_unlearn = parameter_ori - ihvp.numpy()
    parameter_unlearn_ = parameter_ori - ihvp.numpy() * len(dataset_train) / len(dataset_remain)
    print("inf scaled: ", np.linalg.norm(parameter_unlearn - parameter_retrain, ord = np.inf))
    print("inf unscaled: ", np.linalg.norm(parameter_unlearn_ - parameter_retrain, ord = np.inf))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-u', '--unlearn_prop', type=float)
    # parser.add_argument('-m', '--model', type=str, help='affine or nn', choices=['affine', 'nn'])
    # args = parser.parse_args()

    # with open("config.json") as f:
    #     config = json.load(f)
    
    # main(args, config)

    main()