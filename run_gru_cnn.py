import pandas as pd
import torch
from torch import nn, transpose
from torch.utils.data import DataLoader
import torch
import pandas as pd
from model import GRU_CNN
from loaddata import CycloneDataset
import numpy as np
from utils import moving_average, MSELoss_denorm
from train import train, evaluate_denorm

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



    n_epochs = 20
    input_size_ra = 56  # compute eachtime when convNet changes its kernel size and num_channels
    input_size_meta = 15
    hidden_size = 32
    output_size = 6
    num_layers = 5
    batch_size = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_dataset = CycloneDataset(train_meta, train_ra, train_labels, device)
    val_dataset = CycloneDataset(test_meta, test_ra, test_labels, device)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    model = GRU_CNN(input_size_ra, input_size_meta, hidden_size, output_size, num_layers,device,use_ra = True).to(device)
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    loss_fn = nn.MSELoss()

    train(n_epochs, model, train_loader, val_loader, optimizer, loss_fn, lr_scheduler, init_h = True)
    model_loss, model_loss_ts = evaluate_denorm(model, val_loader, loss_fn, init_h = True)
    print("Finel GRY MSE denormed over all timestep: {} \nFinel GRU MSE denormed for each timestep: {}".format(model_loss, model_loss_ts))

    #calculate the baseline MSE (normalized)
    test_moving_avg = moving_average(test_meta)
    norm_MSE = MSELoss_denorm(output = test_moving_avg, label = test_labels, denorm = False, time_sep = True)

    denormed_MSE = MSELoss_denorm(output = test_moving_avg, label = test_labels, denorm = True, time_sep = True)

    print("Over all timestep: \nBaseline normed MSE: {} \nBaseline denormed MSE: {}"
        .format(MSELoss_denorm(output = test_moving_avg, label = test_labels, denorm = False, time_sep = False), 
                MSELoss_denorm(output = test_moving_avg, label = test_labels, denorm = True, time_sep = False)))

    print("By each timestep: \nBaseline normed MSE: {} \nBaseline denormed MSE: {}".format(norm_MSE, denormed_MSE))




    
    
    
    # metric_fn = nn.MSELoss()

    # for epoch in range(1, n_epochs + 1):
    #     print('epoch:',epoch)
    #     loss = train_one_epoch(epoch, train_loader, model, optimizer, lr_scheduler, loss_fn, log_period=100)
    #     metric, metric_ma = validate_ma(val_loader, model, metric_fn)
    #     print('epoch:{}/{}, Train Loss = {}, Validation metric = {}, metric_moving_average = {}'.format(epoch, n_epoch, loss, metric, metric_ma))
    #     print(mse_validation(val_loader, model, metric_fn))
