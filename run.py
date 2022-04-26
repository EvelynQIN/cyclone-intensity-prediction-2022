import torch
import pandas as pd
from sklearn.metrics import mean_squared_error
from model import TCN, ConvNet
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
    model = TCN(input_channels, output_size, channel_sizes, kernel_size=kernel_size, dropout=dropout)

    # Optimization operation: Adam 
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # transform data into tensor
    train_data = CycloneDataset(train_meta, train_ra, train_labels)
    test_data = CycloneDataset(test_meta, test_ra, test_labels)   

    # Create DataLoaders 
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    train(n_epochs, model, train_loader,val_loader, optimizer, loss_fn)
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


    # test dimension of cnn_concat
    # model = ConvNet([8, 16], [2, 2], activation=torch.nn.ReLU())  
    # iter = 0
    # for ( _ , ra_features), label in train_loader:
        
    #     if iter < 2:
    #         model(ra_features)
    #         iter += 1
    #     else:
    #         break
    

    # iter = 0
    # for feature, _ in train_loader:
        
    #     if iter < 2:
    #         model(feature)
    #         iter += 1
    #     else:
    #         break



