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
from train import train, evaluate_denorm

BATCH_SIZE = 128
NUM_LAYERS = 10

if __name__ == '__main__':
    train_path = os.path.join('datasets', 'train_N_extra')
    test_path = os.path.join('datasets', 'test')
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
    
    n_epochs = 10

    input_channels = 71  # calculate based on the CNN setting
    output_size = 6

    # Note: We use a very simple setting here (assuming all levels have the same # of channels.
    channel_sizes = [8] * 3 # [num of hidden units per layer] * num of levels
    kernel_size = 3
    dropout = 0.2

    hidden_size = 4
    num_layers = 2

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    print("Using {} to process".format(device))
    
    train_dataset = CycloneDataset(train_meta, train_ra, train_labels, device)
    val_dataset = CycloneDataset(val_meta, val_ra, val_labels, device)
    test_dataset = CycloneDataset(test_meta, test_ra, test_labels, device)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    model = TCN_GRU(input_channels, output_size, channel_sizes, kernel_size, dropout, hidden_size, num_layers, device).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    loss_fn = nn.MSELoss()

    train(n_epochs, model, train_loader, val_loader, optimizer, loss_fn, 'GRU_TCN', lr_scheduler, init_h = True)

  
    model_loss, model_loss_ts = evaluate_denorm(model, test_loader, loss_fn, scaler_dict, init_h = True)
    print("====================Compare on test set:===============================")
    print("Finel GRU_TCN MSE denormed over all timestep: {} \nFinel GRU_TCN MSE denormed for each timestep: {}".format(model_loss, model_loss_ts))

    #calculate the baseline MSE (normalized)
    test_moving_avg = moving_average(test_meta)
    norm_MSE = MSELoss_denorm(output = test_moving_avg, label = test_labels, scaler_dict = scaler_dict, denorm = False, time_sep = True)

    denormed_MSE = MSELoss_denorm(output = test_moving_avg, label = test_labels, scaler_dict = scaler_dict, denorm = True, time_sep = True)

    print("Over all timestep: \nBaseline normed MSE: {} \nBaseline denormed MSE: {}"
        .format(MSELoss_denorm(output = test_moving_avg, label = test_labels, scaler_dict = scaler_dict, denorm = False, time_sep = False), 
                MSELoss_denorm(output = test_moving_avg, label = test_labels, scaler_dict = scaler_dict, denorm = True, time_sep = False)))

    print("By each timestep: \nBaseline normed MSE: {} \nBaseline denormed MSE: {}".format(norm_MSE, denormed_MSE))