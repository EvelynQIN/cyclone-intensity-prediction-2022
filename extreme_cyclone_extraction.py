
import pandas as pd
from dataset import extract_extreme_cyclones
from sklearn.metrics import mean_squared_error
import pickle
import os


if __name__ == '__main__':

    to_trainpath = os.path.join('datasets', "train_N_extra_normal")
    to_testpath = os.path.join('datasets', "test_N_extra_extreme")

    if not os.path.exists(to_trainpath):
        os.makedirs(to_trainpath)
    if not os.path.exists(to_testpath):
        os.makedirs(to_testpath)

    # # extract trainset & val set
    extract_extreme_cyclones(
                        raw_path = "predictors",
                        ra_feature_names = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320'],
                        meta_feature_names = ['pmin', 'x', 'y', 'z', 'month'],
                        start_stride = 3,
                        split = True, 
                        split_ratio = 0.88,
                        year_range = [2000, 2009], # test_data
                        extreme = False,
                        months_list = ['01','02', '03','04','05','06','07','08','09','10','11','12'],
                        train_path = to_trainpath, 
                        test_path = None,
                        tropical = 'extra',
                        hemi = 'N'
                    )
    print("***********Data extraction complete*************")

    # # extract testset (standardize based on trainset)
    extract_extreme_cyclones(
                            raw_path = "predictors",
                            ra_feature_names = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320'],
                            meta_feature_names = ['pmin', 'x', 'y', 'z', 'month'],
                            start_stride = 1,
                            split = False, 
                            split_ratio = 0.88,
                            year_range = [2000, 2009], # test_data
                            extreme = True,
                            months_list = ['01','02', '03','04','05','06','07','08','09','10','11','12'],
                            train_path = to_trainpath, 
                            test_path = to_testpath,
                            tropical = 'extra',
                            hemi = 'N'
                        )
    # print("***********Test Data extraction complete*************")

    
 
    

    


    



