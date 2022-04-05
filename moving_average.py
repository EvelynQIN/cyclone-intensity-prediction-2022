import numpy as np
def moving_average(meta_features, labels):
    X_intensity = meta_features[:, :, 1]
    pred_intensity = []
    for i in range(len(X_intensity)):
        list_temp = list(X_intensity[i])
        for j in range(7):
            temp_pred = np.mean(list_temp[j:j+7])
            list_temp.append(temp_pred)
        pred_intensity.append(list_temp[7:])
    
    pred_intensity = np.vstack(pred_intensity)

    true_intensity = labels.reshape(-1, labels.shape[1])

    return pred_intensity