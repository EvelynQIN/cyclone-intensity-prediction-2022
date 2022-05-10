import torch
import pandas as pd
from sklearn.metrics import mean_squared_error
from model import CNNTest
from loaddata import CycloneDataset
from train import train, evaluate_denorm
import numpy as np
from utils import moving_average, MSELoss_denorm


if __name__ == '__main__':

    # load the test toy dataset
    train_labels = pd.read_pickle("datasets/train_labels.pkl")
    test_labels = pd.read_pickle("datasets/test_labels.pkl")
    train_meta = pd.read_pickle("datasets/train_meta.pkl")
    test_meta = pd.read_pickle("datasets/test_meta.pkl")
    train_ra = pd.read_pickle("datasets/train_ra.pkl")
    test_ra = pd.read_pickle("datasets/test_ra.pkl")
    moving_avg = pd.read_pickle("datasets/moving_avg.pkl")
    labels = pd.read_pickle("datasets/labels.pkl")
    scaler_dict = pd.read_pickle("datasets/scaler_dict.pkl")


    INIT_LR = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 100
    GROUPS = 7
    PREDICT_STEP = 7
    # input dim for nn
    INPUT_DIM = 90

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    input_channels = 79  # calculate based on the CNN setting
    output_size = 6
    batch_size = 128
    seq_length = 6
    n_epochs = 30
    loss_fn = torch.nn.MSELoss()

    # Note: We use a very simple setting here (assuming all levels have the same # of channels.
    channel_sizes = [8] * 3 # [num of hidden units per layer] * num of levels
    kernel_size = 3
    dropout = 0.2

    model = CNNTest(groups=GROUPS, predict_step=PREDICT_STEP).to(device) # need params!

    # Optimization operation: Adam 
    learning_rate = 1e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    # transform data into tensor
    train_data = CycloneDataset(train_meta, train_ra, train_labels, device)
    test_data = CycloneDataset(test_meta, test_ra, test_labels, device)   

    # Create DataLoaders 
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    train(n_epochs, model, train_loader,val_loader, optimizer, loss_fn, lr_scheduler)
    model_loss, model_loss_ts = evaluate_denorm(model, val_loader, loss_fn)
    print("Finel TCN MSE denormed over all timestep: {} \nFinel TCN MSE denormed for each timestep: {}".format(model_loss, model_loss_ts))

    #calculate the baseline MSE (normalized)
    test_moving_avg = moving_average(test_meta)
    norm_MSE = MSELoss_denorm(output = test_moving_avg, label = test_labels, denorm = False, time_sep = True)

    denormed_MSE = MSELoss_denorm(output = test_moving_avg, label = test_labels, denorm = True, time_sep = True)

    print("Over all timestep: \nBaseline normed MSE: {} \nBaseline denormed MSE: {}"
        .format(MSELoss_denorm(output = test_moving_avg, label = test_labels, denorm = False, time_sep = False), 
                MSELoss_denorm(output = test_moving_avg, label = test_labels, denorm = True, time_sep = False)))

    print("By each timestep: \nBaseline normed MSE: {} \nBaseline denormed MSE: {}".format(norm_MSE, denormed_MSE))