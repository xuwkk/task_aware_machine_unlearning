"""
Train the neural network based load forecasting
We always train an accuracy-driven load forecaster and unlearn from the accuracy-driven load forecaster as well.
We first train the nn on the core dataset, and use the sensitive dataset to fine tune the last layer of the nn.
"""

import torch
from torch.nn.functional import mse_loss
import hydra
from omegaconf import DictConfig
from time import time
from torch.utils.data import DataLoader
from func_operation import return_nn_model, return_core_datasets
import os
from utils import set_random_seed, return_dataset



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
                loss_sum += loss.item() * len(target)
        
        return loss_sum / len(self.testloader.dataset)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    batch_size = cfg.data.batch_size
    epochs = cfg.model.epochs
    lr = cfg.model.lr
    dataset_choice = cfg.model.dataset_choice

    set_random_seed(cfg.data.random_seed)

    """
    generate the dataset    
    """
    dataset_train, dataset_test = return_dataset(cfg)
    
    if dataset_choice == "core":
        dataset_core, dataset_sensitive = return_core_datasets(cfg, dataset_train)
        dataset_train = dataset_core
    elif dataset_choice == "all":
        pass
    else:
        raise Exception("dataset not found!")

    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
    loader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = False)

    # net
    net = return_nn_model(cfg, is_load = False)

    if dataset_choice == "sensitive" or dataset_choice == "remain":
        # currently we did not fine tune using the stochastic method
        net = return_nn_model(cfg, is_load = True, dataset = "core") # load the core dataset trained model
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
    print('no. of parameters: ', no_param)
    
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    trainer = Trainer_MSE(net, optimizer, loader_train, loader_test)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max =  int(epochs/10), eta_min = 0.01 * lr)

    save_dir = cfg.model.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_dir += f'{cfg.model.dataset_choice}.pth'
    
    best_loss = 1e5

    for i in range(1, epochs+1):
        start_time = time()
        train_loss = trainer.train()
        test_loss = trainer.eval()
        
        print("Epoch {}: train loss-{:.4f}, test loss-{:.4f}({:.4f}), time-{:.2f}".format(i, train_loss, test_loss, best_loss, time() - start_time))
        lr_scheduler.step()
        for param_group in trainer.optimizer.param_groups:
            print("LR: {:.6f}".format(param_group['lr']))
        
        if cfg.model.watch == "test":
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(trainer.net.state_dict(), save_dir)
                print("Best model saved!")
        elif cfg.model.watch == "train":
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(trainer.net.state_dict(), save_dir)
                print("Best model saved!")
        else:
            raise Exception("watch not found!")
        
        print("==============================================")

if __name__ == '__main__':
    
    main()
    
    
    # parser = argparse.ArgumentParser()
    # # core and all are trained on the nn model. sensitive is fine tuning the last layer
    # parser.add_argument('-d', '--dataset', type=str, help = "choose from core and all")
    # parser.add_argument('-w', '--watch', type=str)
    # args = parser.parse_args()