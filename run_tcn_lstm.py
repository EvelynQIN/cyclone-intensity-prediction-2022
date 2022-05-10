import pandas as pd
import torch
from torch import nn, transpose
from torch.utils.data import DataLoader
import torch
import pandas as pd
from model import TCN_LSTM
from loaddata import CycloneDataset
import numpy as np
from utils import moving_average, MSELoss_denorm
from train import train, evaluate_denorm


TRAIN_TEST_SPLIT_RATIO = (0.8, 0.2)
BATCH_SIZE = 64
NUM_LAYERS = 10

if __name__ == '__main__':
    train_labels = pd.read_pickle("datasets/train_labels.pkl")
    test_labels = pd.read_pickle("datasets/test_labels.pkl")
    train_meta = pd.read_pickle("datasets/train_meta.pkl")
    test_meta = pd.read_pickle("datasets/test_meta.pkl")
    train_ra = pd.read_pickle("datasets/train_ra.pkl")
    test_ra = pd.read_pickle("datasets/test_ra.pkl")
    moving_avg = pd.read_pickle("datasets/moving_avg.pkl")
    labels = pd.read_pickle("datasets/labels.pkl")
    scaler_dict = pd.read_pickle("datasets/scaler_dict.pkl")



    n_epochs = 10

    input_channels = 79  # calculate based on the CNN setting
    output_size = 6
    # Note: We use a very simple setting here (assuming all levels have the same # of channels.
    channel_sizes = [8] * 3 # [num of hidden units per layer] * num of levels
    kernel_size = 3
    dropout = 0.2

    lstm_hidden_size = 4
    num_layers = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    train_dataset = CycloneDataset(train_meta, train_ra, train_labels)
    val_dataset = CycloneDataset(test_meta, test_ra, test_labels)

    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = False)

    model = TCN_LSTM(input_channels, output_size, channel_sizes, kernel_size, dropout, lstm_hidden_size, num_layers).to(device)
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    loss_fn = nn.MSELoss()

    train(device, n_epochs, model, train_loader, val_loader, optimizer, loss_fn, lr_scheduler, init_h = True)
    model_loss, model_loss_ts = evaluate_denorm(model, val_loader, loss_fn, init_h = True)
    print("Finel TCN MSE denormed over all timestep: {} \nFinel TCN MSE denormed for each timestep: {}".format(model_loss, model_loss_ts))

    #calculate the baseline MSE (normalized)
    test_moving_avg = moving_average(test_meta)
    norm_MSE = MSELoss_denorm(output = test_moving_avg, label = test_labels, denorm = False, time_sep = True)

    denormed_MSE = MSELoss_denorm(output = test_moving_avg, label = test_labels, denorm = True, time_sep = True)

    print("Over all timestep: \nBaseline normed MSE: {} \nBaseline denormed MSE: {}"
        .format(MSELoss_denorm(output = test_moving_avg, label = test_labels, denorm = False, time_sep = False), 
                MSELoss_denorm(output = test_moving_avg, label = test_labels, denorm = True, time_sep = False)))

    print("By each timestep: \nBaseline normed MSE: {} \nBaseline denormed MSE: {}".format(norm_MSE, denormed_MSE))