
import pandas as pd
from dataset import extract_timeseries
from sklearn.metrics import mean_squared_error
import pickle
import os


if __name__ == '__main__':

    to_trainpath = "train_N_extra"
    to_testpath = "test"

    if not os.path.exists(os.path.join('datasets', to_trainpath)):
        os.makedirs(os.path.join('datasets', to_trainpath))
    if not os.path.exists(os.path.join('datasets', to_testpath)):
        os.makedirs(os.path.join('datasets', to_testpath))

    # extract trainset & val set
    extract_timeseries(
                        raw_path = "predictors",
                        ra_feature_names = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320'],
                        meta_feature_names = ['pmin', 'x', 'y', 'z', 'month'],
                        start_stride = 3,
                        split = True,
                        split_ratio = 0.88,
                        year_range = [2000, 2009], # train_data
                        months_list = ['01','02','03','04','05','06','07','08','09','10','11','12'],
                        train_path = to_trainpath, 
                        test_path = None,
                        tropical = 'extra',
                        hemi = 'N'
                              )
    print("***********Data extraction complete*************")

    # extract testset (standardize based on trainset)
    extract_timeseries(
                        raw_path = "predictors",
                        ra_feature_names = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320'],
                        meta_feature_names = ['pmin', 'x', 'y', 'z', 'month'],
                        start_stride = 1,
                        split = False, 
                        split_ratio = 0.88,
                        year_range = [2009, 2010], # test_data
                        months_list = ['01','02','03','04','05','06','07','08','09','10','11','12'],
                        train_path = to_trainpath, 
                        test_path = to_testpath,
                        tropical = 'extra',
                        hemi = 'N'
                    )
    print("***********Test Data extraction complete*************")

    # extract trainset & val set for test run. --> just 2 months of the first year
    # to_trainpath = "datasets/train_toy"
    
    # extract_timeseries(
    #                     raw_path = "predictors",
    #                     ra_feature_names = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320'],
    #                     meta_feature_names = ['pmin', 'x', 'y', 'z', 'month'],
    #                     start_stride = 1,
    #                     split = True,
    #                     split_ratio = 0.88,
    #                     year_range = [2000, 2001], # train_data
    #                     months_list = ['01','02'],
    #                     train_path = to_trainpath, 
    #                     test_path = None,
    #                     tropical = 'extra',
    #                     hemi = 'N'
    #                           )

    
 
    

    


    



