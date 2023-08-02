"""
generate the unlearning indexes based on their impact on the test loss: mse, mape, and cost
1. trained affine model
2. trained nn model
"""

import json
from torch.utils.data import DataLoader
from utils.dataset import return_dataset
from func_operation import return_trained_model, return_module, return_dataset_for_nn_affine, return_core_datasets
import numpy as np
import time
import sys
from tqdm import tqdm

save_dir = 'influence'
model_type_list = ["affine", "nn"]
creteria_list = ["mse", "mape", "cost"]

with open("config.json") as f:
    config = json.load(f)

case_name = "case14"
batch_size = config['batch_size_eval']
shuffle = False

for model_type in model_type_list:
    
    train_loss = config[model_type]['train_loss']
    print("model type: {}, train loss: {}".format(model_type, train_loss))
    
    dataset_train, dataset_test = return_dataset(case_name, model_type = model_type)
    
    if model_type == "nn":
        dataset_core, dataset_sensitive = return_core_datasets(dataset_to_be_split=dataset_train)
        dataset_train, dataset_test = return_dataset_for_nn_affine(dataset_sensitive, dataset_test)
    
    print("number of training samples: ", len(dataset_train), "number of test samples: ", len(dataset_test))
    print('feature shape: ', dataset_train.feature.shape, 'target shape: ', dataset_train.feature.shape)
    
    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = shuffle)
    loader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = shuffle)
    
    # calculate the gradient on the train set
    model_train = return_trained_model(model_type = model_type + "-affine" if model_type == "nn" else model_type, 
                                    dataset_train = dataset_train, is_spo = False)
    
    no_param = sum(p.numel() for p in model_train.parameters() if p.requires_grad)
    print("no of parameters: ", no_param)
    
    module_train = return_module(loss_type_dict={"train": train_loss, "test": "mse"}, # "test" here is not used
                            loader_dict={"train": loader_train, "test": None},
                            model_type=model_type, model=model_train, method = 'cg', watch_progress = False)
    
    start_train = time.time()
    grad_train_all = []
    for i in tqdm(range(len(dataset_train)), total = len(dataset_train), desc = "calculating train grad"):
        grad_train_all.append(module_train.train_loss_grad(train_idxs=[i]).numpy())
    
    print("time for calculating train grad: ", time.time() - start_train)
    print("shape of the train grad: ", grad_train_all[0].shape)
    print("max train grad: ", np.max(np.abs(np.mean(grad_train_all))))
    
    for creteria in creteria_list:
        print("creteria loss: ", creteria)
        
        # generate average gradient on the test set on the creteria metric
        if creteria != "cost":
            model_test = return_trained_model(model_type = model_type + "-affine" if model_type == "nn" else model_type, 
                                            dataset_train = dataset_train, is_spo = False)
        else:
            model_test = return_trained_model(model_type = model_type + "-affine" if model_type == "nn" else model_type, 
                                            dataset_train = dataset_train, is_spo = True)
        
        module_test = return_module(loss_type_dict={"train": "mse", "test": creteria}, # train here is not used
                            loader_dict={"train": loader_train, "test": loader_test},
                            model_type=model_type, model=model_test, method = 'cg')
        
        start_time = time.time()
        grad_test_ave = module_test.test_loss_grad(test_idxs=range(len(dataset_test)))
        print("time for calculating test grad: ", time.time() - start_time)
        
        
        # calculate M matrix defined in the paper; we do not use spo which is too slow
        start_time = time.time()
        M = -module_train.inverse_hvp(vec = grad_test_ave).numpy()
        print("time for calculating M: ", time.time() - start_time)
        
        # calculate the influence
        influences = []
        for grad in grad_train_all:
            influences.append(grad @ M)
        
        scale = -1
        influences = scale * np.array(influences) / len(dataset_train) # ! average the influence
        
        np.save(file = f"{save_dir}/{model_type}_{creteria}.npy", arr = influences)
        print("====================================")