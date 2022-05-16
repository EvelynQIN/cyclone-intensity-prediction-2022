import numpy as np
import pandas as pd
from pandas.compat.pickle_compat import _LoadSparseSeries
from sklearn.metrics import mean_squared_error
import torch
from torch.cuda.memory import list_gpu_processes

def moving_average(meta_features):
        X_intensity = meta_features[:, :, 0]
        pred_intensity = []
        for i in range(len(X_intensity)):
            list_temp = list(X_intensity[i])
            for j in range(6):
                temp_pred = np.mean(list_temp[j:j+6])
                list_temp.append(temp_pred)
            pred_intensity.append(list_temp[6:])
        
        pred_intensity = np.vstack(pred_intensity)

        return pred_intensity

def MSELoss_denorm(output, label, scaler_dict, denorm = False, time_sep = False):
    """ denormed MSE loss
        Args:
            time_sep: True then calculate MSE for each time step
            denorm: if True then denorm the label
    """
    if denorm:
        scaler_pmin = scaler_dict['pmin']
        output = output * scaler_pmin[1] + scaler_pmin[0]
        label = label * scaler_pmin[1] + scaler_pmin[0]

    if time_sep:
        MSE = []
        for t in range(output.shape[1]): # iter over all timestep
            MSE.append(mean_squared_error(output[:,t], label[:,t]))

    else:
        
        MSE = mean_squared_error(output, label)

    return MSE
  
class Weighted_MSELoss(torch.nn.Module):
  '''Wrapper class for L1 loss that set different weight to each time step'''

  def __init__(self, weight_list):
    super().__init__()
    self.weight_list = weight_list

  def forward(self, output, label):
    loss = 0
    for t in range(output.shape[1]):
      loss += torch.mean(torch.pow((output[:, t] - label[:, t]), 2)) * self.weight_list[t]
    return loss
