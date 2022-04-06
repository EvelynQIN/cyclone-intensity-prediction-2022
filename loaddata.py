import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class Transformer:

    def __init__(self, data_dir):

        self.rng = np.random.RandomState(42)
        self.ra_feature_file = os.path.join(data_dir, "ra_features.pkl")
        self.meta_file = os.path.join(data_dir, "meta_features.pkl")
        self.label_file = os.path.join(data_dir, "labels.pkl")

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

        self.one_hot_encoder()


    def ra_features_clamper(self):        
        ra_features = pd.read_pickle(self.ra_feature_file)

        # Iterate through each re-analysis feature and clamp each feature
        for feature_name in self.ra_clamp_values:
            # Clamp the features
            fmin, fmax, indices = self.ra_clamp_values[feature_name]
            ra_features[:, :, :, indices] = np.clip(ra_features[:, :, :, indices], a_min=fmin, a_max=fmax)
        self.ra_features = ra_features

    def one_hot_encoder(self):
        meta_features = pd.read_pickle(self.meta_file)

        month_matrix = meta_features[:, :, 3]

        meta_features = np.delete(meta_features, 3, 2)

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
    
    def train_test_split(self, ratio = 0.88, norm = True):
        """ split train and test based on ratio and perform standard scaler
        Args:
            ratio: split ratio
            norm: perform standard scaler if True
        Returns:
            train_labels, test_labels, train_meta, test_meta, train_ra, test_ra
        """
        self.labels = pd.read_pickle(self.label_file)
        datasize = self.labels.shape[0]
        split_ind = int(datasize * ratio)
        print("train size: {}   ||    test size {}".format(split_ind, datasize - split_ind))
        train_labels, test_labels = self.labels[:split_ind], self.labels[split_ind:]
        train_meta, test_meta = self.meta_features[:split_ind], self.meta_features[split_ind:]
        train_ra, test_ra = self.ra_features[:split_ind], self.ra_features[split_ind:]

        if norm == True:

            # check standardize 逻辑， 一直报错，而且 time, location坐标还有month是不是不需要做标准化？

            # meta_train_shape = train_meta.shape
            # train_meta = train_meta.reshape((meta_train_shape[0], -1))
            # scaler = StandardScaler().fit(train_meta)
            # train_meta = scaler.transform(train_meta)
            # train_meta = train_meta.reshape(meta_train_shape)

            # meta_test_shape = test_meta.shape
            # test_meta = test_meta.reshape((meta_test_shape[0], -1))
            # test_meta = scaler.transform(test_meta)
            # test_meta = test_meta.reshape(meta_test_shape)

            # scaler = StandardScaler().fit(train_labels)
            # train_labels = scaler.transform(train_labels)
            # test_labels = scaler.transform(test_labels)

            # ra_train_shape = train_ra.shape
            # train_ra = train_ra.reshape((ra_train_shape[0], -1))
            # scaler = StandardScaler().fit(train_ra)
            # train_ra = scaler.transform(train_ra)
            # train_ra = train_ra.reshape(ra_train_shape)

            # ra_test_shape = test_ra.shape
            # test_ra = test_ra.reshape((ra_test_shape[0], -1))
            # test_ra = scaler.transform(test_ra)
            # test_ra = test_ra.reshape(test_ra)
            print("做的不对，之后再改 ：)")
        
        return train_labels, test_labels, train_meta, test_meta, train_ra, test_ra

class CycloneDataset(torch.utils.data.Dataset):

    def __init__(self, feature_NN, feature_CNN, target):
        super().__init__()
        self.feature_NN = torch.from_numpy(feature_NN).float()
        self.feature_CNN = torch.from_numpy(feature_CNN).float()

        self.target = torch.from_numpy(target).float()  


    def __getitem__(self, index):
        (feature_NN, feature_CNN), target = (self.feature_NN[index], self.feature_CNN[index]), self.target
        return (feature_NN, feature_CNN), target

    def __len__(self):
        return self.feature_NN.size(0)







    
    