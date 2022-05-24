
import pandas as pd
from dataset import extract_timeseries
from sklearn.metrics import mean_squared_error
import pickle
import os


if __name__ == '__main__':

    to_trainpath = os.path.join('datasets', "train_N_tropical")
    to_testpath = os.path.join('datasets', "test_N_tropical")

    if not os.path.exists(to_trainpath):
        os.makedirs(to_trainpath)
    if not os.path.exists(to_testpath):
        os.makedirs(to_testpath)

    # extract trainset & val set
    extract_timeseries(
                        raw_path = "predictors",
                        ra_feature_names = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320'],
                        meta_feature_names = ['pmin', 'x', 'y', 'z', 'month'],
                        start_stride = 1,
                        split = True,
                        split_ratio = 0.88,
                        year_range = [2000, 2009], # train_data
                        months_list = ['01','02','03','04','05','06','07','08','09','10','11','12'],
                        train_path = to_trainpath, 
                        test_path = None,
                        tropical = 'tropical',
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
                        tropical = 'tropical',
                        hemi = 'N'
                    )
    print("***********Test Data extraction complete*************")

    
 
    

    


    



