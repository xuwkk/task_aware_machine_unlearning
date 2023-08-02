"""
evaluate the direct unlearning performance on
1. mape
2. cost
of the following models
1. affine model
2. reverse affine model
3. nn model

Three options for generating the unlearning set
construct the unlearning set based on the choice of {random, helpful, harmful}
1. helpful: if removed, the test set mape will be significantly increased.
2. harmful: if removed, the test set mape will be significantly decreased.
"""

import json
import argparse
from torch.utils.data import DataLoader
from utils.dataset import return_dataset
from utils.funcs import evaluate, reconstruct_model, flatten_model
from func_operation import return_unlearn_datasets, return_trained_model, return_module, return_core_datasets, return_dataset_for_nn_affine
import numpy as np
import torch
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--unlearn_prop', type=float)
parser.add_argument('-m', '--mode', type=str, help='how to choose the unlearn set: helpful or harmful or random', default = 'helpful')
parser.add_argument('--model', type=str, help='affine or nn')
args = parser.parse_args()

with open("config.json") as f:
    config = json.load(f)

unlearn_prop = args.unlearn_prop
batch_size = config['batch_size_eval']
case_name = "case14"
model_type = args.model
influence_dir = config[args.model]['influence_dir']
print(influence_dir)
save_dir = f"simulation_result/{model_type}/"
train_loss = config[model_type]['train_loss']

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

"""
generate the dataset
"""
influences = torch.from_numpy(np.load(influence_dir)).float()
dataset_train, dataset_test = return_dataset(case_name, model_type = model_type)
if model_type == "nn":
    dataset_core, dataset_sensitive = return_core_datasets(dataset_to_be_split=dataset_train)
    dataset_train, dataset_test = return_dataset_for_nn_affine(dataset_sensitive, dataset_test)

dataset_unlearn, dataset_remain, unlearn_index, remain_index = return_unlearn_datasets(influences=influences, 
                                                        unlearn_prop=unlearn_prop, dataset_to_be_unlearn=dataset_train, 
                                                        mode=args.mode)

loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = False)
loader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = False)
loader_unlearn = DataLoader(dataset_unlearn, batch_size = batch_size, shuffle = False)
loader_remain = DataLoader(dataset_remain, batch_size = batch_size, shuffle = False)

print('Load is scaled:\n', dataset_train.is_scale, dataset_test.is_scale, dataset_unlearn.is_scale, dataset_remain.is_scale)
print('train dataset shape: ', dataset_train.feature.shape, 'test dataset shape: ', dataset_test.feature.shape)


"""
construct the model
"""
model_ori = return_trained_model(model_type = model_type + "-affine" if model_type == "nn" else model_type,
                                dataset_train = dataset_train, is_spo = False)
model_retrain = return_trained_model(model_type = model_type + "-affine" if model_type == "nn" else model_type, 
                                    dataset_train = dataset_remain, is_spo = False)

model_ori.eval()
model_retrain.eval()
parameter_ori = flatten_model(model_ori)
parameter_retrain = flatten_model(model_retrain)

print(parameter_ori.shape, parameter_retrain.shape)

"""
machine unlearning
"""

# unlearn using the default model
module = return_module(
    loss_type_dict = {"train":train_loss, "test": train_loss}, # "test" here is not used
    loader_dict={"train": loader_remain, "test": loader_remain},
    model_type=model_type, model=model_ori, method = 'cg', watch_progress=False
)

ihvp = module.stest(test_idxs=range(len(dataset_remain))).numpy()
# print("ihvp: ", ihvp)

parameter_unlearn = parameter_ori - ihvp
model_unlearn = reconstruct_model(model = model_ori, flattened_params = parameter_unlearn)

print('max difference between unlearn and retrain: ', np.max(np.abs(parameter_retrain - parameter_unlearn)))

mse_ori = evaluate(model_ori, {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = 'mse')
mape_ori = evaluate(model_ori, {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = 'mape')
cost_ori = evaluate(model_ori, {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = 'cost')

print('original model:')
print("mse:\n", mse_ori, "\nmape:\n", mape_ori, "\ncost:\n", cost_ori)

mse_unlearn = evaluate(model_unlearn, {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = 'mse')
mape_unlearn = evaluate(model_unlearn, {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = 'mape')
cost_unlearn = evaluate(model_unlearn, {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = 'cost')

print('unlearned model:')
print("mse:\n", mse_unlearn, "\nmape:\n", mape_unlearn, "\ncost:\n", cost_unlearn)

mse_retrain = evaluate(model_retrain, {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = 'mse')
mape_retrain = evaluate(model_retrain, {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = 'mape')
cost_retrain = evaluate(model_retrain, {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = 'cost')

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




# np.save(save_dir + "mse_ori.npy", mse_ori)
# np.save(save_dir + "mape_ori.npy", mape_ori)
# np.save(save_dir + "cost_ori.npy", cost_ori)
# np.save(save_dir + "mse_unlearn.npy", mse_unlearn)
# np.save(save_dir + "mape_unlearn.npy", mape_unlearn)
# np.save(save_dir + "cost_unlearn.npy", cost_unlearn)
# np.save(save_dir + "mse_retrain.npy", mse_retrain)
# np.save(save_dir + "mape_retrain.npy", mape_retrain)
# np.save(save_dir + "cost_retrain.npy", cost_retrain)







# over_gen_index = np.where(cost_ori["test_gen_mismatch"] > 0)[0]
# under_gen_index = np.where(cost_ori["test_gen_mismatch"] <= 0)[0]

# print("no of over gen: ", len(over_gen_index), "no of under gen: ", len(under_gen_index))
# print("over gen average mse: ", mse_ori['test'][over_gen_index].mean(), "under gen average mse: ", mse_ori['test'][under_gen_index].mean())

# select_mse_index = mse_ori['test'][under_gen_index].argsort()[:len(over_gen_index)] 

# plt.figure()
# plt.scatter(x = mse_ori['test'][over_gen_index], y = cost_ori['test'][over_gen_index], marker = 'o', color = 'red', label = 'over gen')
# plt.scatter(x = mse_ori['test'][under_gen_index][select_mse_index], y = cost_ori['test'][under_gen_index][select_mse_index], marker = 'o', color = 'green', label = 'under gen')
# plt.xlabel('mse')
# plt.ylabel('cost')
# plt.legend()
# plt.legend()
# plt.savefig('figs/{}_mse-cost.png'.format(args.model))
# plt.show()

# plt.figure()
# plt.scatter(x = mse_ori['test'][over_gen_index], y = mape_ori['test'][over_gen_index], marker = 'o', color = 'red', label = 'over gen')
# plt.scatter(x = mse_ori['test'][under_gen_index][select_mse_index], y = mape_ori['test'][under_gen_index][select_mse_index], marker = 'o', color = 'green', label = 'under gen')
# plt.xlabel('mse')
# plt.ylabel('mape')
# plt.legend()
# plt.legend()
# plt.savefig('figs/{}_mse-mape.png'.format(args.model))
# plt.show()

# plt.figure()
# plt.scatter(x = mape_ori['test'][over_gen_index], y = cost_ori['test'][over_gen_index], marker = 'o', color = 'red', label = 'over gen')
# plt.scatter(x = mape_ori['test'][under_gen_index][select_mse_index], y = cost_ori['test'][under_gen_index][select_mse_index], marker = 'o', color = 'green', label = 'under gen')
# plt.xlabel('mape')
# plt.ylabel('cost')
# plt.legend()
# plt.legend()
# plt.savefig('figs/{}_mape-cost.png'.format(args.model))
# plt.show()
