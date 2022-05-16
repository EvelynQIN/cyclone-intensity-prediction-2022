
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle


def train_step(model, train_loader, optimizer, loss_fn, lr_scheduler = None, init_h = None):
    """
    Train model for 1 epoch.
    """

    model.train()

    for features, label in train_loader:
        # features, label = (features[0].to(device), features[1].to(device)), label # put the data on the selected execution device
        optimizer.zero_grad()   # zero the parameter gradients

        if init_h == True:
            h = model.init_hidden(features[1].shape[0])
            output = model(features, h)
        else:
            output = model(features)  # forward pass

        loss = loss_fn(output, label)    # compute loss
        loss.backward() # backward pass

        optimizer.step()    # perform update
    
    return loss

def evaluate(model, val_loader, loss_fn, init_h = None):
    """
    Evaluate model on validation data.
    """
    model.eval()

    total_loss = 0
    
    with torch.no_grad():
        for features, label in val_loader:
            # features, label = (features[0].to(device), features[1].to(device)), label.to(device) # put the data on the selected execution device
            if init_h == True:
                h = model.init_hidden(features[1].shape[0])
                output = model(features, h)
            else:
                output = model(features)   # forward pass
            loss = loss_fn(output, label)    # compute loss

            total_loss += loss.item()
        
    total_loss /= len(val_loader)

    return total_loss

def train(n_epochs, model, train_loader,val_loader, optimizer, loss_fn, lr_scheduler = None, init_h = None):
    """
    Train and evaluate model.
    """
    loss_dict = dict()
    loss_dict['train'] = []
    loss_dict['val'] = []

    for epoch in range(n_epochs):
        
        # train model for one epoch
        train_loss = train_step(model, train_loader, optimizer, loss_fn, lr_scheduler, init_h)
        loss_dict['train'].append(train_loss)
        # evaluate 
        val_loss = evaluate(model, val_loader, loss_fn, init_h)
        loss_dict['val'].append(val_loss)

        print(f"[Epoch {epoch}] - Training : loss = {train_loss}", end=" ")
        print(f"Validation : loss = {val_loss}")  
    
    pickle.dump(loss_dict, open(f"datasets/model_logs/loss_dict.pkl", "wb"))



def evaluate_denorm(model, val_loader, loss_fn, scaler_dict, init_h = None):
    """
    Evaluate model on validation data.
    """
    model.eval()

    total_loss = 0
    total_loss_ts = [0] * 6

    scaler_pmin = scaler_dict['pmin']
    
    with torch.no_grad():
        
        for features, label in val_loader:
            # features, label = (features[0].to(device), features[1].to(device)), label.to(device) # put the data on the selected execution device
            if init_h == True:
                h = model.init_hidden(features[1].shape[0])
                output = model(features, h)
            else:
                output = model(features)   # forward pass

            output = output * scaler_pmin[1] + scaler_pmin[0]
            label = label * scaler_pmin[1] + scaler_pmin[0]

            loss = loss_fn(output, label)    # compute loss

            total_loss += loss.item()
            
            losses_ts = []
            for t in range(label.shape[1]):
                losst = loss_fn(output[:, t], label[:, t]).item()
                losses_ts.append(losst)
            total_loss_ts = np.add(total_loss_ts, losses_ts)

    total_loss /= len(val_loader)
    total_loss_ts /= len(val_loader)

    return total_loss, total_loss_ts



        






