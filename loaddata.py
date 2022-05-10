import os
import pandas as pd
import numpy as np
import torch


class Transformer:
    """ Perform ra_feature clip, month one hot encoding, standard scaler, and train_test split function
    """

    def __init__(self, data_dir):

        self.rng = np.random.RandomState(42)
        self.ra_feature_file = os.path.join(data_dir, "ra_features.pkl")
        self.meta_file = os.path.join(data_dir, "meta_features.pkl")
        self.label_file = os.path.join(data_dir, "labels.pkl")
        self.scaler_dict_file = os.path.join(data_dir, "scaler_dict.pkl")

        self.ra_clamp_values = {
            # Each feature stores a min, max and indices tuple
            'U300': [-100, 100.0, 0],
            'V300': [-100, 100.0, 1],
            'U500': [-100, 100.0, 2],
            'V500': [-100, 100.0, 3],
            'T850': [200, 330, 4],
            'MSL': [8e4, 1.2e5, 5],
            'PV320': [-30, 30.0, 6]
        }

        self.ra_features_clamper()

        self.one_hot_encoder(month_index = 4)

        self.standard_scaler()


    def ra_features_clamper(self):        
        ra_features = pd.read_pickle(self.ra_feature_file)

        # Iterate through each re-analysis feature and clamp each feature
        for feature_name in self.ra_clamp_values:
            # Clamp the features
            fmin, fmax, indices = self.ra_clamp_values[feature_name]
            ra_features[:, :, :, indices] = np.clip(ra_features[:, :, :, indices], a_min=fmin, a_max=fmax)
        self.ra_features = ra_features

    def one_hot_encoder(self, month_index):
        """Append 11 0-1 digits to the tail of feature vectors
        """
        meta_features = pd.read_pickle(self.meta_file)

        month_matrix = meta_features[:, :, month_index]

        meta_features = np.delete(meta_features, month_index, 2)

        months = [i for i in range(1, 13)]
        one_encodes = pd.get_dummies(months, drop_first=True)
        month_encoder = dict()        
        for month in months:
            month_encoder[month] = np.array(one_encodes.loc[month-1])
        
        rows = np.array([])
        for row in range(meta_features.shape[0]):
            cols = np.array([])
            for col in range(meta_features.shape[1]):
                cols = np.append(cols, [np.concatenate((meta_features[row, col], month_encoder[month_matrix[row, col]]))], axis = 0) if cols.size > 0 else np.array([np.concatenate((meta_features[row, col], month_encoder[month_matrix[row, col]]))])
            rows = np.append(rows, [cols], axis =0) if rows.size > 0 else np.array([cols])
        self.meta_features = rows
    
    def standard_scaler(self, cols = 4):
        """ Perform standard scaler on all features and labels
        Args:
            cols: num of cols to be scaled in the meta features
        """
        scaler_dict = pd.read_pickle(self.scaler_dict_file)
        labels = pd.read_pickle(self.label_file)

        # standardize labels
        self.labels = (labels - scaler_dict['pmin'][0]) / scaler_dict['pmin'][1]

        # standardize meta featrues --> first 4 cols ['pmin', 'x', 'y', 'z', 'month-one-hot']
        meta_cols = ['pmin', 'x', 'y', 'z']
        for i in range(cols):
            self.meta_features[:, :, i] = (self.meta_features[:, :, i] - scaler_dict[meta_cols[i]][0]) / scaler_dict[meta_cols[i]][1]

        # standardize ra featrues --> ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320']
        ra_cols = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320']
        for i in range(len(ra_cols)):
            self.ra_features[:, :, :, i] = (self.ra_features[:, :, :, i] - scaler_dict[ra_cols[i]][0]) / scaler_dict[ra_cols[i]][1]

    
    def train_test_split(self, ratio = 0.88):
        """ split train and test based on ratio and perform standard scaler
        Args:
            ratio: split ratio
            norm: perform standard scaler if True
        Returns:
            train_labels, test_labels, train_meta, test_meta, train_ra, test_ra
        """
        datasize = self.labels.shape[0]
        split_ind = int(datasize * ratio)
        print("train size: {}   ||    test size {}".format(split_ind, datasize - split_ind))
        train_labels, test_labels = self.labels[:split_ind], self.labels[split_ind:]
        train_meta, test_meta = self.meta_features[:split_ind], self.meta_features[split_ind:]
        train_ra, test_ra = self.ra_features[:split_ind], self.ra_features[split_ind:]
        
        return train_labels, test_labels, train_meta, test_meta, train_ra, test_ra

class CycloneDataset(torch.utils.data.Dataset):

    def __init__(self, meta_feature, ra_feature, label, device):
        super().__init__()
        self.meta_feature = torch.from_numpy(meta_feature).float()

        # expected ra features to be (sample_size * time_steps * channels * 11 * 11)
        self.ra_feature = torch.from_numpy(ra_feature.transpose((0, 1, 3, 2)).reshape(-1, 6, 7, 11, 11)).float()

        self.label = torch.from_numpy(label).float()  
      
        self.device = device


    def __getitem__(self, index):
        (meta_feature, ra_feature), label = (self.meta_feature[index], self.ra_feature[index]), self.label[index]
        return (meta_feature.to(self.device), ra_feature.to(self.device)), label.to(self.device)

    def __len__(self):
        return self.meta_feature.size(0)







    
    