import pandas as pd
import torch
from torch import nn, transpose
from torch.utils.data import DataLoader
import os
import pandas as pd
from model import TCN_GRU
from loaddata import CycloneDataset
import numpy as np
from utils import moving_average, MSELoss_denorm
from train import train, evaluate_denorm, evaluate

BATCH_SIZE = 64
n_epochs = 100
input_channels = 79  # calculate based on the CNN setting
output_size = 6
# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [64] * 3 # [num of hidden units per layer] * num of levels
kernel_size = 3
dropout = 0.2
hidden_size = 128
num_layers = 8
learning_rate = 1e-5


if __name__ == '__main__':
    train_path = 'datasets/train_N_tropical'
    test_path = 'datasets/test_N_tropical'
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    train_labels = pd.read_pickle(train_path + "/train_labels.pkl")
    val_labels = pd.read_pickle(train_path + "/val_labels.pkl")
    train_meta = pd.read_pickle(train_path + "/train_meta.pkl")
    val_meta = pd.read_pickle(train_path + "/val_meta.pkl")
    train_ra = pd.read_pickle(train_path + "/train_ra.pkl")
    val_ra = pd.read_pickle(train_path + "/val_ra.pkl")
    scaler_dict = pd.read_pickle(train_path + "/scaler_dict.pkl")

    test_labels = pd.read_pickle(test_path + "/labels.pkl")  
    test_meta = pd.read_pickle(test_path + "/meta_features.pkl")    
    test_ra = pd.read_pickle(test_path + "/ra_features.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    print("Using {} to process".format(device))
    
    train_dataset = CycloneDataset(train_meta, train_ra, train_labels, device)
    val_dataset = CycloneDataset(val_meta, val_ra, val_labels, device)
    test_dataset = CycloneDataset(test_meta, test_ra, test_labels, device)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    model = TCN_GRU(input_channels, output_size, channel_sizes, kernel_size, dropout, hidden_size, num_layers, device).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)  
    
    # decay the learning rate when reaching milestones
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
    loss_fn = nn.MSELoss()

    train(n_epochs, model, train_loader, val_loader, optimizer, loss_fn, 'GRU_TCN_N_tropical', lr_scheduler, init_h = True)

  
    denorm_model_loss, denorm_model_loss_ts = evaluate_denorm(model, test_loader, loss_fn, scaler_dict, init_h = True)
    model_loss, model_loss_ts = evaluate(model, test_loader, loss_fn, init_h = True)
    print("====================Compare on test set:===============================")
    print(f"Over all timestep: \n GRU_TCN normed MSE: {model_loss} \n GRU_TCN denormed MSE {denorm_model_loss}")
    print(f"By each timestep: \n GRU_TCN normed MSE: {model_loss_ts} \n GRU_TCN denormed MSE: {denorm_model_loss_ts}")

    #calculate the baseline MSE (normalized)
    # test_moving_avg = moving_average(test_meta)
    # norm_MSE = MSELoss_denorm(output = test_moving_avg, label = test_labels, scaler_dict = scaler_dict, denorm = False, time_sep = True)

    # denormed_MSE = MSELoss_denorm(output = test_moving_avg, label = test_labels, scaler_dict = scaler_dict, denorm = True, time_sep = True)

    # print("Over all timestep: \nBaseline normed MSE: {} \nBaseline denormed MSE: {}"
    #     .format(MSELoss_denorm(output = test_moving_avg, label = test_labels, scaler_dict = scaler_dict, denorm = False, time_sep = False), 
    #             MSELoss_denorm(output = test_moving_avg, label = test_labels, scaler_dict = scaler_dict, denorm = True, time_sep = False)))

    # print("By each timestep: \nBaseline normed MSE: {} \nBaseline denormed MSE: {}".format(norm_MSE, denormed_MSE))