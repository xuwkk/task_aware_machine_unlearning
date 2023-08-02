"""
check the results of 
1. linear model
2. linear model upon the trained neutal network
"""

import json
import argparse
from torch.utils.data import DataLoader
from utils.dataset import return_dataset
from utils.funcs import flatten_model
from func_operation import return_unlearn_datasets, return_trained_model, return_module, return_core_datasets, return_dataset_for_nn_affine
import numpy as np
import torch
import sys
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--unlearn_prop', type=float)
parser.add_argument('--model', type=str, help='affine or nn')
args = parser.parse_args()

with open("config.json") as f:
    config = json.load(f)

model_type = args.model
batch_size = config['batch_size_eval']
case_name = "case14"
train_loss = config[model_type]['train_loss']

"""
generate the dataset: random unlearning
"""
unlearn_prop = args.unlearn_prop
dataset_train, dataset_test = return_dataset(case_name, model_type = model_type)

if model_type == "nn":
    dataset_core, dataset_sensitive = return_core_datasets(dataset_to_be_split=dataset_train)
    dataset_train, dataset_test = return_dataset_for_nn_affine(dataset_sensitive, dataset_test)

dataset_unlearn, dataset_remain, unlearn_index, remain_index = return_unlearn_datasets(influences=None, 
                                                        unlearn_prop=unlearn_prop, dataset_to_be_unlearn=dataset_train, 
                                                        mode="random")

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
test the gradient on the train dataset
"""
module = return_module(loss_type_dict={'train':'mse', 'test':'mse'}, 
                    loader_dict = {'train': loader_train, 'test': loader_train},
                    model_type=model_type, model = model_ori, method = 'cg')

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
module = return_module(loss_type_dict={'train':'mse', 'test':'mse'},
                    loader_dict = {'train': loader_remain, 'test': loader_remain},
                    model_type=model_type, model = model_ori, method = 'cg', watch_progress = False)

ihvp = module.stest(test_idxs = range(len(dataset_remain)))
parameter_unlearn = parameter_ori - ihvp.numpy()

print('the max difference between parameters: ', np.max(np.abs(parameter_retrain - parameter_unlearn)))
assert np.isclose(parameter_unlearn, parameter_retrain, atol=1e-4).all(), "parameter should be the same"
print("===============================================")

# test direct
print("Test direct")
module = return_module(loss_type_dict={'train':'mse', 'test':'mse'},
                    loader_dict = {'train': loader_remain, 'test': loader_remain},
                    model_type=model_type, model = model_ori, method = 'direct')

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
module = return_module(loss_type_dict={'train':'mse', 'test':'mse'},
                    loader_dict = {'train': loader_remain, 'test': loader_unlearn},
                    model_type=model_type, model = model_ori, method = 'cg')

ihvp = module.stest(test_idxs = range(len(dataset_unlearn))) * len(dataset_unlearn) / len(dataset_remain)
parameter_unlearn = parameter_ori + ihvp.numpy()
print(  'the max difference between parameters: ', np.max(np.abs(parameter_retrain - parameter_unlearn)))
assert np.isclose(parameter_unlearn, parameter_retrain, atol=1e-4).all(), "parameter should be the same"

print("===============================================")
"""
unlearning result comparison
    hessian: train
    gradient: remain
"""
print("Unlearning performance. Hessian: train, grad: remain.")
module = return_module(loss_type_dict={'train':'mse', 'test':'mse'},
                    loader_dict = {'train': loader_train, 'test': loader_remain},
                    model_type=model_type, model = model_ori, method = 'cg')
ihvp = module.stest(test_idxs = range(len(dataset_remain)))
parameter_unlearn = parameter_ori - ihvp.numpy()
parameter_unlearn_ = parameter_ori - ihvp.numpy() * len(dataset_train) / len(dataset_remain)
print("inf scaled: ", np.linalg.norm(parameter_unlearn - parameter_retrain, ord = np.inf))
print("inf unscaled: ", np.linalg.norm(parameter_unlearn_ - parameter_retrain, ord = np.inf))



# """
# construct the model
# """
# dataset_collect = {'train': dataset_train, 'test': dataset_test, 'remain': dataset_remain, 'unlearn': dataset_unlearn}
# if model_type == 'affine':
#     # use analytic solver
#     analytic_solver = AnalyticSolverAffine(dataset_collect = dataset_collect)
#     parameter_analytic = analytic_solver.fit_regressor(mode = 'train')
    
#     # convert to torch nn model
#     linear_model = ModelAffine(parameter_analytic)
# else:
#     # use both sklearn solver and analytic solver
#     sklearn_solver = SklearnSolver(dataset_collect=dataset_collect)
#     regressor_train = sklearn_solver.fit_regressor(mode = 'train')
#     analytic_solver = AnalyticSolverReverseAffine(dataset_collect=dataset_collect)
#     parameter_analytic = analytic_solver.fit_regressor(mode = 'train')
#     weight_analytic = parameter_analytic[:, :-1]
#     bias_analytic = parameter_analytic[:, -1]
    
#     # check if the parameters are the same
#     # print(np.linalg.norm(regressor_train.coef_ - weight_analytic, ord = np.inf) )
#     # print(np.linalg.norm(regressor_train.intercept_ - bias_analytic, ord = np.inf) )
#     assert np.allclose(regressor_train.coef_, weight_analytic, atol = 1e-3)
#     assert np.allclose(regressor_train.intercept_, bias_analytic, atol = 1e-5)
#     print('weight shape: {}, bias shape: {}'.format(regressor_train.coef_.shape, regressor_train.intercept_.shape))
    
#     linear_model = ModelReverseAffine(regressor_train)
    
# linear_model.eval()

# mape = evaluate(linear_model, dataset_collection = dataset_collect, mode = 'mape')

# print("mape of original model:\n", mape)
# print("===============================================")

# """
# test the gradient on the train set (should be close to zero)
# """
# print('gradient on train set')
# grad_train = analytic_solver.cal_grad(mode = 'train', parameter = parameter_analytic)
# print('l1 norm of the train gradient using analytic_solver: ',  np.linalg.norm(grad_train, ord = 1))

# module = return_module(loss_type_dict={'train':'mse', 'test':'mse'}, loader_dict = {'train': loader_train, 'test': loader_train}, 
#                     model = linear_model, method = 'cg')
# grad_train = module.test_loss_grad(test_idxs = range(len(dataset_train))) # test loss test loader
# print('l1 norm of the train gradient using torch-influence: ',  torch.norm(grad_train, p = 1))

# module = return_module(loss_type_dict={'train':'mse', 'test':'mse'}, loader_dict = {'train': loader_remain, 'test': loader_unlearn}, 
#                     model = linear_model, method = 'cg')
# grad_unlearn_sum = module.test_loss_grad(test_idxs = range(len(dataset_unlearn))) * len(dataset_unlearn)
# grad_remain_sum = module.train_loss_grad(train_idxs = range(len(dataset_remain))) * len(dataset_remain)
# print('l1 norm of the sum of the remain and unlearn gradient using torch-influence: ',  torch.norm((grad_unlearn_sum + grad_remain_sum) / len(dataset_train), p = 1))

# print("===============================================")

# """
# test the gradient of the module and analytic solver

# For the reverse affine model
# 1. the gradient calculated from module is concatenated as a vector 
# 2. (weight.flatten, bias) (88, 88, ... 88, 14), same as hessian
# """
# grad_analytic = analytic_solver.cal_grad(mode = 'unlearn', parameter = parameter_analytic)
# module = return_module(loss_type_dict={'train':'mse', 'test':'mse'}, 
#                         loader_dict = {'train': loader_train, 'test': loader_unlearn}, 
#                         model = linear_model, 
#                         method = 'direct')
# grad_module = module.test_loss_grad(test_idxs = range(len(dataset_unlearn))) # test loss test loader

# print('shape of the gradients: ', grad_analytic.shape, grad_module.shape)
# print('max difference between analytic and module gradient: ', np.max(np.abs(grad_analytic - grad_module.numpy())))

# print("===============================================")

# """
# test the hessian of module and analytic solver
# """
# hessian_analytic = analytic_solver.cal_hessian(mode = 'train', parameter = parameter_analytic)
# hessian_module = module.hessian

# print('shape of the hessian: ', hessian_analytic.shape, hessian_module.shape)
# # print(hessian_analytic[0])
# # print(hessian_module.numpy()[0])
# print('max difference between analytic and module hessian: ', np.max(np.abs(hessian_analytic - hessian_module.numpy())))

# print("===============================================")

# """
# unlearning result comparison
#     hessian: remain
#     gradient: remain
# """
# print("Unlearning performance. Hessian: remain, grad: remain.")
# parameter_retrain = analytic_solver.fit_regressor(mode = 'remain')
# parameter_ori = analytic_solver.fit_regressor(mode = 'train')

# # flatten the reverse affine model
# if model_type == 'reverse-affine':
#     parameter_retrain = np.concatenate((parameter_retrain[:,:-1].flatten(), parameter_retrain[:,-1]))
#     parameter_ori = np.concatenate((parameter_ori[:,:-1].flatten(), parameter_ori[:,-1]))

# module = return_module(loss_type_dict={'train':'mse', 'test':'mse'}, 
#                     loader_dict = {'train': loader_remain, 'test': loader_remain}, 
#                     model = linear_model, method = 'cg')

# # using stest function provided by the module
# # the stest calculated the hessian based on train set (train loss) and gradient based on test set (test loss)
# # ! all have been averaged by their number of samples
# ihvp = module.stest(test_idxs = range(len(dataset_remain)))
# parameter_unlearn = parameter_ori - ihvp.numpy()
# print('the max difference between parameters: ', np.max(np.abs(parameter_retrain - parameter_unlearn)))
# assert np.isclose(parameter_retrain, parameter_unlearn, atol=1e-3).all(), "parameter should be the same"
# print('The stest is correct and the unlearning using cg is correct')

# # unlearn using the analytic solver
# grad = analytic_solver.cal_grad(mode = 'remain', parameter = parameter_analytic)
# hessian = analytic_solver.cal_hessian(mode = 'remain', parameter = parameter_analytic)
# ihvp = np.linalg.inv(hessian) @ grad
# parameter_unlearn = parameter_ori - ihvp
# print('max difference between parameters: ', np.max(np.abs(parameter_retrain - parameter_unlearn)))
# assert np.isclose(parameter_retrain, parameter_unlearn, atol=1e-4).all(), "parameter should be the same"
# print("The analytic unlearning is correct")

# # unlearn using the module direct
# module = return_module(loss_type_dict={'train':'mse', 'test':'mse'}, loader_dict = {'train': loader_remain, 'test': loader_remain}, model = linear_model, method = 'direct') 
# ihvp = module.stest(test_idxs = range(len(dataset_remain)))
# parameter_unlearn = parameter_ori - ihvp.numpy()
# print('max difference between parameters: ', np.max(np.abs(parameter_retrain - parameter_unlearn)))
# assert np.isclose(parameter_retrain, parameter_unlearn, atol=1e-3).all(), "parameter should be the same"
# print("The direct unlearning is correct")

# print("===============================================")

# """
# unlearning
#     hessian: remain
#     gradient: unlearn
#     this is also an exact unlearning
# """

# print("Unlearning performance. Hessian: remain, grad: unlearn.")
# module = return_module(loss_type_dict={'train':'mse', 'test':'mse'}, loader_dict = {'train': loader_remain, 'test': loader_unlearn}, model = linear_model, method = 'cg')
# # rescale the gradient
# # ! the gradient calculated from module is averaged by the number of samples (unlearn)
# ihvp = module.stest(test_idxs = range(len(dataset_unlearn))) * len(dataset_unlearn) / len(dataset_remain)
# parameter_unlearn = parameter_ori + ihvp.numpy() # ! here should be plus
# print('max difference between parameters: ', np.max(np.abs(parameter_retrain - parameter_unlearn)))
# assert np.isclose(parameter_retrain, parameter_unlearn, atol=1e-3).all(), "parameter should be the same"
# print('The unlearning using unlearn dataset as gradient is correct')

# print("===============================================")

# """
# unlearning
#     hessian: train
#     gradient: remain
#     not exact unlearning
# """
# print("Unlearning performance. Hessian: train, grad: remain.")
# module = return_module(loss_type_dict={'train':'mse', 'test':'mse'}, loader_dict = {'train': loader_train, 'test': loader_remain}, model = linear_model, method = 'cg')
# # find the hessian on the train dataloader and gradient on the test dataloader
# # ! all averaged based on their size, the same to the paper setting
# ihvp = module.stest(test_idxs = range(len(dataset_remain)))
# ihvp_noscale = ihvp * len(dataset_remain) / len(dataset_train)
# parameter_unlearn = parameter_ori - ihvp.numpy()
# parameter_unlearn_noscale = parameter_ori - ihvp_noscale.numpy()

# print('diff with rescale (better as in paper): ', np.linalg.norm(parameter_unlearn - parameter_retrain, ord = 2))
# print('diff without rescale (worse): ', np.linalg.norm(parameter_unlearn_noscale - parameter_retrain, ord = 2))
