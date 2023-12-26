"""
generate the unlearning indexes based on their impact on the test loss: mse, mape, and cost
    the influence os each training samples to the average performance on the test set (like the regular influence function used to the find the influence)
    formula: expected_test_loss_grad * inverse_hessian_of_entire_train * grad_of_train_sample
1. trained linear model
2. trained nn_conv model (on the last linear layer)
3. trained nn_mlpmixer model (on the last linear layer)
"""

from torch.utils.data import DataLoader
from utils import return_dataset
from func_operation import return_trained_model, return_module, return_dataset_for_nn_affine, return_core_datasets
import numpy as np
import time
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    save_dir = cfg.influence_dir
    criteria_list = ["mse", "mape", "cost"]  # the metrics to evaluate the inlfuence on the test set performance
    batch_size = cfg.data.batch_size
    model_type = cfg.model.type
    train_loss = cfg.model.train_loss
    shuffle = False

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("model type: {}, train loss: {}".format(model_type, train_loss))
    
    dataset_train, dataset_test = return_dataset(cfg)

    if 'nn' in model_type:
        # split the train set into core and sensitive sets
        dataset_core, dataset_sensitive = return_core_datasets(cfg, dataset_to_be_split=dataset_train)
        # dataset for training the last layer of nn
        dataset_train, dataset_test = return_dataset_for_nn_affine(cfg, dataset_sensitive, dataset_test)

    print("number of training samples: ", len(dataset_train), "number of test samples: ", len(dataset_test))
    print('feature shape: ', dataset_train.feature.shape, 'target shape: ', dataset_train.feature.shape)

    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = shuffle)
    loader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = shuffle)

    # the trained linear model
    model_train = return_trained_model(cfg, model_type = model_type + "-affine" if 'nn' in model_type else model_type, 
                                        dataset_train = dataset_train, is_spo = False)
    
    no_param = sum(p.numel() for p in model_train.parameters() if p.requires_grad)
    print("no of parameters: ", no_param)

    # influence func module
    module_train = return_module(cfg, 
                                loss_type_dict={"train": train_loss, "test": "mse"},      # "test" here is not used as we only calculate the gradient on the train set
                                loader_dict={"train": loader_train, "test": None},
                                model=model_train, 
                                method = 'cg', 
                                watch_progress = False)
    
    # calculate the gradient of each sample in the train set
    start_train = time.time()
    grad_train_all = []
    for i in tqdm(range(len(dataset_train)), total = len(dataset_train), desc = "calculating train grad"):
        grad_train_all.append(module_train.train_loss_grad(train_idxs=[i]).numpy())
    
    print("time for calculating train grad: ", time.time() - start_train)
    print("shape of the train grad: ", grad_train_all[0].shape)
    print("max train grad: ", np.max(np.abs(np.mean(grad_train_all))))

    for criteria in criteria_list:

        print("====================================")
        print("creteria loss: ", criteria)
        
        # generate average gradient on the test set on the criteria metric
        if criteria != "cost":
            model_test = return_trained_model(cfg,
                                            model_type = model_type + "-affine" if 'nn' in model_type else model_type, 
                                            dataset_train = dataset_train, is_spo = False)
        else:
            model_test = return_trained_model(cfg,
                                            model_type = model_type + "-affine" if 'nn' in model_type else model_type, 
                                            dataset_train = dataset_train, is_spo = True)
        
        module_test = return_module(cfg,
                                    loss_type_dict={"train": "mse", "test": criteria},
                                    loader_dict={"train": loader_train, "test": loader_test},
                                    model=model_test, 
                                    method = 'cg')
        
        start_time = time.time()
        grad_test_ave = module_test.test_loss_grad(test_idxs=range(len(dataset_test)))
        print("time for calculating test grad: ", time.time() - start_time)
        
        # calculate m matrix defined in the paper
        start_time = time.time()
        M = -module_train.inverse_hvp(vec = grad_test_ave).numpy()  # todo: why dont use stest?
        print("time for calculating M: ", time.time() - start_time)
        
        # calculate the influence
        influences = []
        for grad in grad_train_all:
            influences.append(grad @ M)
        
        scale = -1
        influences = scale * np.array(influences) / len(dataset_train) # ! average the influence
        
        np.save(file = f"{save_dir}/{model_type}_{criteria}.npy", arr = influences)



if __name__ == "__main__":
    main()