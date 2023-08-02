"""
train the neural network based load forecasting
1. mse driven
2. spo driven
"""

import torch
from torch.nn.functional import mse_loss

class Trainer:
    
    def __init__(self, net, optimizer, train_loader, test_loader):
        
        self.net = net
        self.optimizer = optimizer
        self.trainloader = train_loader
        self.testloader = test_loader

class Trainer_MSE(Trainer):
        
    def train(self):
        self.net.train()
        loss_sum = 0.
        for feature, target in self.trainloader:
            self.optimizer.zero_grad()
            output = self.net(feature)[1] # 1 represents the output of the predictor
            # loss = torch.mean(torch.abs(output - target) / target)
            loss = mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item() * len(target)
            
        return loss_sum / len(self.trainloader.dataset)
    
    def eval(self):
        self.net.eval()
        loss_sum = 0.
        with torch.no_grad():
            for feature, target in self.testloader:
                output = self.net(feature)[1] # 1 represents the output of the predictor
                loss = mse_loss(output, target)
                # loss = torch.mean(torch.abs(output - target) / target)
                loss_sum += loss.item() * len(target)
        
        return loss_sum / len(self.testloader.dataset)
    
    
if __name__ == '__main__':
    
    from time import time
    import json
    from utils.dataset import return_dataset
    from torch.utils.data import DataLoader
    import numpy as np
    import random
    from func_operation import return_nn_model, return_core_datasets, return_unlearn_datasets
    import os
    import argparse
    
    parser = argparse.ArgumentParser()
    # core and all are trained on the nn model. sensitive is fine tuning the last layer
    parser.add_argument('-d', '--dataset', type=str, help = "choose from core and all")
    parser.add_argument('-w', '--watch', type=str)
    args = parser.parse_args()
    
    with open("config.json") as f:
        config = json.load(f)
    
    case_name = "case14"
    model_type = "nn"
    random_seed = config['random_seed']
    batch_size = config['batch_size']
    epochs = config['nn']['epochs']
    lr = config['nn']['lr']
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    """
    generate the dataset    
    """
    dataset_train, dataset_test = return_dataset(case_name, model_type = model_type)
    
    if args.dataset == "core":
        dataset_core, dataset_sensitive = return_core_datasets(dataset_train)
        dataset_train = dataset_core
    elif args.dataset == "all":
        pass
    else:
        raise Exception("dataset not found!")
    
    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
    loader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = False)
    
    # net
    net = return_nn_model(is_load = False)
    if args.dataset == "sensitive" or args.dataset == "remain":
        net = return_nn_model(is_load = True, dataset = "core") # load the core dataset trained model
        # freeze the parameters of the core dataset trained model and only train the last layer
        for name, param in net.named_parameters():
            if not "lin2" in name:
                param.requires_grad = False
        
    print("trainable parameters:")
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
    
    no_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print('train dataset shape: ', dataset_train.feature.shape, 'test dataset shape: ', dataset_test.feature.shape)
    print('is scaled: ', dataset_test.is_scale, dataset_train.is_scale)
    print('No. of parameters: ', no_param)
    
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    trainer = Trainer_MSE(net, optimizer, loader_train, loader_test)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max =  int(epochs/10), eta_min = 0.01 * lr)

    best_loss = 1e5
    save_path = 'trained_model/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path += f'nn_{args.dataset}.pth'
    
    for i in range(1, epochs+1):
        start_time = time()
        train_loss = trainer.train()
        test_loss = trainer.eval()
        
        print("Epoch {}: train loss-{:.4f}, test loss-{:.4f}({:.4f}), time-{:.2f}".format(i, train_loss, test_loss, best_loss, time() - start_time))
        lr_scheduler.step()
        for param_group in trainer.optimizer.param_groups:
            print("LR: {:.6f}".format(param_group['lr']))
        
        if args.watch == "test":
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(trainer.net.state_dict(), save_path)
                print("Best model saved!")
        elif args.watch == "train":
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(trainer.net.state_dict(), save_path)
                print("Best model saved!")
        
        print("==============================================")