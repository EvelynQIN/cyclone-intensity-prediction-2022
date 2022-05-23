import pandas as pd
import torch
from torch import nn, transpose
from torch.utils.data import DataLoader
from model import GRU_CNN
from loaddata import CycloneDataset
import numpy as np
from utils import MSELoss_denorm
from train import train, evaluate_denorm, evaluate

if __name__ == '__main__':
    train_path = 'datasets/train_N_extra'
    test_path = 'datasets/test_N_extra'

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



    n_epochs = 5
    input_size_ra = 64  # compute eachtime when convNet changes its kernel size and num_channels
    input_size_meta = 15
    hidden_size = 64
    output_size = 6
    num_layers = 6
    BATCH_SIZE = 64
    learning_rate = 1e-5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_dataset = CycloneDataset(train_meta, train_ra, train_labels, device)
    val_dataset = CycloneDataset(val_meta, val_ra, val_labels, device)
    test_dataset = CycloneDataset(test_meta, test_ra, test_labels, device)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    model = GRU_CNN(input_size_ra, input_size_meta, hidden_size, output_size, num_layers,device,use_ra = True).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)  
    
    # decay the learning rate when reaching milestones
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

    loss_fn = nn.MSELoss()

    train(n_epochs, model, train_loader, val_loader, optimizer, loss_fn, 'GRU_CNN_1', lr_scheduler, init_h = True)
    
    
    # # load test data ==================================================
    # test_path = 'datasets/test_N_extra'
    # train_path = 'datasets/train_N_extra'
    # test_labels = pd.read_pickle(test_path + "/labels.pkl")  
    # test_meta = pd.read_pickle(test_path + "/meta_features.pkl")    
    # test_ra = pd.read_pickle(test_path + "/ra_features.pkl")
    # scaler_dict = pd.read_pickle(train_path + "/scaler_dict.pkl")

    # BATCH_SIZE = 64
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loss_fn = nn.MSELoss()

    # print("Using {} to process".format(device))

    # # flatten the ra features for shap explainer
    # test_dataset = CycloneDataset(test_meta, test_ra, test_labels, device)

    # test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    # # load the model to explain
    # save_path = 'datasets/checkpoints/GRU_CNN_100.tar'
    # checkpoint = torch.load(save_path, map_location="cuda")
    # args = checkpoint['args']

    # model = GRU_CNN(**args).to("cuda")
    # model.load_state_dict(checkpoint['model_state_dict'])   
    
    denorm_model_loss, denorm_model_loss_ts = evaluate_denorm(model, test_loader, loss_fn, scaler_dict, init_h = True)
    model_loss, model_loss_ts = evaluate(model, test_loader, loss_fn, init_h = True)
    print("====================Compare on test set:===============================")
    print(f"Over all timestep: \n GRU_CNN normed MSE: {model_loss} \n GRU_CNN denormed MSE {denorm_model_loss}")
    print(f"By each timestep: \n GRU_CNN normed MSE: {model_loss_ts} \n GRU_CNN denormed MSE: {denorm_model_loss_ts}")


    
    
    

