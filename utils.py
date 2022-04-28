import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

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

def MSELoss_denorm(output, label, denorm = False, time_sep = False):
    """ denormed MSE loss
        Args:
            time_sep: True then calculate MSE for each time step
            denorm: if True then denorm the label
    """
    if denorm:
        scaler_pmin = pd.read_pickle("datasets/scaler_dict.pkl")['pmin']
        output = output * scaler_pmin[1] + scaler_pmin[0]
        label = label * scaler_pmin[1] + scaler_pmin[0]

    if time_sep:
        MSE = []
        for t in range(output.shape[1]): # iter over all timestep
            MSE.append(mean_squared_error(output[:,t], label[:,t]))

    else:
        
        MSE = mean_squared_error(output, label)

    return MSE
