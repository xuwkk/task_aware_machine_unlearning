"""
evaluate the performance unchanged unlearning on the following models
1. affine model
2. reverse affine model
3. nn model

based on the creteria
1. mape unchange
2. cost unchange
"""

import json
import argparse
from torch.utils.data import DataLoader
from utils.dataset import return_dataset, DatasetWithWeight
from utils.funcs import evaluate, reconstruct_model, flatten_model
from func_operation import return_unlearn_datasets, return_trained_model, return_module, return_core_datasets, return_dataset_for_nn_affine
import numpy as np
import torch
import time
import sys
import cvxpy as cp

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--unlearn_prop', type=float)
parser.add_argument('-m', '--mode', type=str, help='how to choose the unlearn set: helpful or harmful or random', default = 'helpful')
parser.add_argument('--model', type=str, help='affine or nn')
parser.add_argument('-c', "--creteria", type=str, help='mse, mape, or cost')
args = parser.parse_args()

with open("config.json") as f:
    config = json.load(f)

unlearn_prop = args.unlearn_prop
batch_size = config['batch_size_eval']
case_name = "case14"
model_type = args.model
influence_dir = config[args.model]['influence_dir']
train_loss = config[args.model]['train_loss']
print("model type: {}, train loss: {}, creteria: {}".format(model_type, train_loss, args.creteria))

l1_constraints = config[model_type]['l1_constraints']

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
# original model
model_ori = return_trained_model(model_type = model_type + "-affine" if model_type == "nn" else model_type,
                                dataset_train = dataset_train, is_spo = False)
model_ori.eval()
parameter_ori = flatten_model(model_ori)

mse_ori = evaluate(model = model_ori,
                dataset_collection= {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = "mse", with_mispatch=True)
mape_ori = evaluate(model = model_ori, 
                dataset_collection= {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = "mape", with_mispatch=True)
cost_ori = evaluate(model = model_ori,
                dataset_collection= {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = "cost", with_mispatch=True)

cost_ori_ = {"remain": cost_ori["remain"], "unlearn": cost_ori["unlearn"], "test": cost_ori["test"]}

print('mse ori: \n', mse_ori)
print('mape ori: \n', mape_ori)
print('cost ori: \n', cost_ori_)


# unlearn model
# unlearn using the default model
module_unlearn = return_module(
    loss_type_dict = {"train":train_loss, "test": train_loss}, # "test" here is not used
    loader_dict={"train": loader_remain, "test": loader_remain},
    model_type=model_type, model=model_ori, method = 'cg', watch_progress=False
)

ihvp = module_unlearn.stest(test_idxs=range(len(dataset_remain))).numpy()
parameter_unlearn_ = parameter_ori - ihvp
model_unlearn = reconstruct_model(model = model_ori, flattened_params = parameter_unlearn_)

mse_unlearn = evaluate(model = model_unlearn,
                dataset_collection= {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = "mse", with_mispatch=True)
mape_unlearn = evaluate(model = model_unlearn,
                dataset_collection= {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = "mape", with_mispatch=True)
cost_unlearn = evaluate(model = model_unlearn,
                dataset_collection= {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = "cost", with_mispatch=True)

cost_unlearn_ = {"remain": cost_unlearn["remain"], "unlearn": cost_unlearn["unlearn"], "test": cost_unlearn["test"]}

print('mse unlearn: \n', mse_unlearn)
print('mape unlearn: \n', mape_unlearn)
print('cost unlearn: \n', cost_unlearn_)

"""
unlearning
"""

# calcualte the gradient on the train dataset
# construct module on train = remain (to calculate the inverse hessian), test = train (to calculate gradient)
module_train = return_module(loss_type_dict={"train": train_loss, "test": train_loss}, 
                        loader_dict={"train": loader_remain, "test": loader_train},
                        # loader_dict={"train": loader_train, "test": loader_train},
                        model_type=model_type, model=model_ori, method = 'cg')

start_train = time.time()
grad_train_all = []
for i in range(len(dataset_train)):
    grad_train_all.append(module_train.test_loss_grad(test_idxs=[i]).numpy())

print("time for calculating train grad: ", round(time.time() - start_train, 2))

# generate average gradient on the test set
if args.creteria != "cost":
    model_test = return_trained_model(model_type = model_type + "-affine" if model_type == "nn" else model_type, 
                                    dataset_train = dataset_train, is_spo = False)
else:
    model_test = return_trained_model(model_type = model_type + "-affine" if model_type == "nn" else model_type, 
                                    dataset_train = dataset_train, is_spo = True)

module_test = return_module(loss_type_dict={"train": "mse", "test": args.creteria}, # train here is not used
                            loader_dict={"train": loader_train, "test": loader_test},
                            model_type=model_type, model=model_test, method = 'cg')

start_time = time.time()
grad_test_ave = module_test.test_loss_grad(test_idxs=range(len(dataset_test)))
print("time for calculating test grad: ", round(time.time() - start_time,2))

# calculate M matrix defined in the paper: hessian: remain
# ! we do not use spo model which is too slow
start_time = time.time()
M = -module_train.inverse_hvp(vec = grad_test_ave).numpy()
print("time for calculating M: ", round(time.time() - start_time, 2))

# calculate the score
scores = []
for grad in grad_train_all:
    scores.append(grad @ M)

scores = np.array(scores) / len(dataset_train) # average over the size of train dataset
scores_remain = scores[remain_index]
scores_unlearn = scores[unlearn_index]
print('performance change of unlearning (positive for helpful): {}'.format(-np.sum(scores_unlearn)))

scores_summary = {
    "remain": scores_remain,
    "unlearn": scores_unlearn,
    "mismatch_remain": cost_ori["remain_gen_mismatch"],
    "mismatch_unlearn": cost_ori["unlearn_gen_mismatch"] 
}

np.save(f'simulation_result/{model_type}/unchange_{unlearn_prop}_{args.creteria}_scores.npy', scores_summary, allow_pickle=True)

# l1_constraint = [start / 2**i for i in range(5)]
# l1_constraint = [0.2, 0.175, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0]
# l1_constraint = [0.25]
# l1_constraint = [0]
linf_constraint = 1.

log = {
    "mse_test_ori": mse_ori["test"],
    "mse_unlearn_ori": mse_ori["unlearn"],
    "mse_remain_ori": mse_ori["remain"],
    "mape_test_ori": mape_ori["test"],
    "mape_unlearn_ori": mape_ori["unlearn"],
    "mape_remain_ori": mape_ori["remain"],
    "cost_test_ori": cost_ori["test"],
    "cost_unlearn_ori": cost_ori["unlearn"],
    "cost_remain_ori": cost_ori["remain"],
    "mse_test_unlearn": mse_unlearn["test"],
    "mse_unlearn_unlearn": mse_unlearn["unlearn"],
    "mse_remain_unlearn": mse_unlearn["remain"],
    "mape_test_unlearn": mape_unlearn["test"],
    "mape_unlearn_unlearn": mape_unlearn["unlearn"],
    "mape_remain_unlearn": mape_unlearn["remain"],
    "cost_test_unlearn": cost_unlearn["test"],
    "cost_unlearn_unlearn": cost_unlearn["unlearn"],
    "cost_remain_unlearn": cost_unlearn["remain"],
    "l1_constraints": l1_constraints,
    "linf_constraint": [linf_constraint] * len(l1_constraints) if isinstance(linf_constraint, float) else linf_constraint,
    "mse_test": [],
    "mse_unlearn": [],
    "mse_remain": [],
    "mape_test": [],
    "mape_unlearn": [],
    "mape_remain": [],
    "cost_test": [],
    "cost_unlearn": [],
    "cost_remain": [],
    "parameter_diff": []
}

for constraint in l1_constraints:
    
    print("========================================================================")
    # print('constraint:', constraint * len(dataset_remain))
    print('constraint:', constraint)

    eps_remain = cp.Variable(len(dataset_remain))
    # obj = cp.Minimize(cp.abs(cp.scalar_product(eps_remain, scores_remain) - np.sum(scores_unlearn)))
    # obj = cp.Minimize(cp.abs(cp.sum(cp.multiply(eps_remain, scores_remain))))
    obj = cp.Minimize(cp.scalar_product(eps_remain, scores_remain)) # we do not need to use abs here
    # average so that can be generalized to different dataset size
    cons = [cp.norm(eps_remain - 1, 1) <= constraint * len(dataset_remain), cp.norm(eps_remain - 1, np.inf) <= linf_constraint]
    prob = cp.Problem(obj, cons)
    prob.solve( verbose=False, solver=cp.GUROBI)
    print('status:', prob.status)
    eps_remain = eps_remain.value
    eps_all = np.zeros(len(dataset_train))
    eps_all[remain_index] = eps_remain
    eps_all[unlearn_index] = -1
    ihvp = 0
    
    # we need to contruct the module so that the weighted mse loss can be applied
    dataset_remain_with_weight = DatasetWithWeight(dataset_remain, eps_remain)
    loader_remain_with_weight = DataLoader(dataset_remain_with_weight, batch_size = batch_size, shuffle = False)
    module_unlearn = return_module(loss_type_dict={"train": train_loss, "test": train_loss}, 
                        # loader_dict={"train": loader_remain, "test": loader_train_with_weight},
                        loader_dict={"train": loader_remain_with_weight, "test": loader_remain_with_weight},
                        model_type=model_type, model=model_ori, method = 'cg', with_weight = True)
    
    # ihvp = module_unlearn.stest(test_idxs=range(len(dataset_train))) * len(dataset_train) / len(dataset_remain)
    ihvp = module_unlearn.stest(test_idxs=range(len(dataset_remain)))
    ihvp = ihvp.numpy()
    parameter_unlearn = parameter_ori - ihvp
    
    # print(ihvp)
    model_unlearn = reconstruct_model(model_ori, parameter_unlearn)
    mse_unlearn = evaluate(model_unlearn, {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = 'mse')
    mape_unlearn = evaluate(model_unlearn, {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = 'mape')
    cost_unlearn = evaluate(model_unlearn, {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}, loss = 'cost')
    
    print(np.linalg.norm(eps_remain - 1, 1), np.linalg.norm(eps_remain - 1, np.inf))
        
    print('mse unlearn: \n', mse_unlearn)
    print('mape unlearn: \n', mape_unlearn)
    print('cost unlearn: \n', cost_unlearn)
    parameter_diff = np.linalg.norm(parameter_unlearn_ - parameter_unlearn, 2)
    print('parameter difference: ', parameter_diff)
    
    log['mse_test'].append(mse_unlearn['test'])
    log['mse_unlearn'].append(mse_unlearn['unlearn'])
    log['mse_remain'].append(mse_unlearn['remain'])
    
    log['mape_test'].append(mape_unlearn['test'])
    log['mape_unlearn'].append(mape_unlearn['unlearn'])
    log['mape_remain'].append(mape_unlearn['remain'])
    
    log['cost_test'].append(cost_unlearn['test'])
    log['cost_unlearn'].append(cost_unlearn['unlearn'])
    log['cost_remain'].append(cost_unlearn['remain'])
    
    log['parameter_diff'].append(parameter_diff)

np.save(f'simulation_result/{model_type}/unchange_{unlearn_prop}_{args.creteria}.npy', log)    


# plt.figure()
# plt.plot(log['l1_constraint'], log['mse_test'], label = 'test', marker = 'o', color = 'red')
# plt.plot(log['l1_constraint'], log['mse_unlearn'], label = 'unlearn', marker = 'o', color = 'green')
# plt.plot(log['l1_constraint'], log['mse_remain'], label = 'remain', marker = 'o', color = 'blue')
# plt.hlines(y = mse_ori['test'], xmin = 0, xmax = max(log['l1_constraint']), label = 'ori-test', linestyle = '--',  color = 'red')
# plt.hlines(y = mse_ori['unlearn'], xmin = 0, xmax = max(log['l1_constraint']), label = 'ori-unlearn', linestyle = '--',    color = 'green')
# plt.hlines(y = mse_ori['remain'], xmin = 0, xmax = max(log['l1_constraint']), label = 'ori-remain', linestyle = '--',   color = 'blue')
# plt.xlabel('l1 constraint')
# plt.ylabel('mse')
# plt.legend()
# plt.savefig('figs/{}_{}_{}-mse.png'.format(args.model, args.creteria, args.unlearn_prop))
# plt.show()

# plt.figure()
# plt.plot(log['l1_constraint'], log['mape_test'], label = 'test', marker = 'o', color = 'red')
# plt.plot(log['l1_constraint'], log['mape_unlearn'], label = 'unlearn', marker = 'o', color = 'green')
# plt.plot(log['l1_constraint'], log['mape_remain'], label = 'remain', marker = 'o', color = 'blue')
# plt.hlines(y = mape_ori['test'], xmin = 0, xmax = max(log['l1_constraint']), label = 'ori-test', linestyle = '--',  color = 'red')
# plt.hlines(y = mape_ori['unlearn'], xmin = 0, xmax = max(log['l1_constraint']), label = 'ori-unlearn', linestyle = '--',    color = 'green')
# plt.hlines(y = mape_ori['remain'], xmin = 0, xmax = max(log['l1_constraint']), label = 'ori-remain', linestyle = '--',   color = 'blue')
# plt.xlabel('l1 constraint')
# plt.ylabel('mape')
# plt.legend()
# plt.savefig('figs/{}_{}_{}-mape.png'.format(args.model, args.creteria, args.unlearn_prop))
# plt.show()

# plt.figure()
# plt.plot(log['l1_constraint'], log['cost_test'], label = 'test', marker = 'o', color = 'red')
# plt.plot(log['l1_constraint'], log['cost_unlearn'], label = 'unlearn', marker = 'o', color = 'green')
# plt.plot(log['l1_constraint'], log['cost_remain'], label = 'remain', marker = 'o', color = 'blue')
# plt.hlines(y = cost_ori['test'], xmin = 0, xmax = max(log['l1_constraint']), label = 'ori-test', linestyle = '--',  color = 'red')
# plt.hlines(y = cost_ori['unlearn'], xmin = 0, xmax = max(log['l1_constraint']), label = 'ori-unlearn', linestyle = '--',    color = 'green')
# plt.hlines(y = cost_ori['remain'], xmin = 0, xmax = max(log['l1_constraint']), label = 'ori-remain', linestyle = '--',   color = 'blue')
# plt.xlabel('l1 constraint')
# plt.ylabel('cost')
# plt.legend()
# plt.savefig('figs/{}_{}_{}-cost.png'.format(args.model, args.creteria, args.unlearn_prop))
# plt.show()

    