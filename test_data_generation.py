
import pandas as pd
from dataset import extract_timeseries
from sklearn.metrics import mean_squared_error
import pickle


if __name__ == '__main__':
<<<<<<< HEAD

    to_trainpath = "datasets/train_N_extra"
    # to_trainpath = "datasets/debugger"

    extract_timeseries(
                        raw_path = "predictors",
                        ra_feature_names = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320'],
                        meta_feature_names = ['pmin', 'x', 'y', 'z', 'month'],
                        start_stride = 3,
                        split = True,
                        split_ratio = 0.88,
                        year_range = [2000, 2009], # train_data
                        months_list = ['01','02','03','04','05','06','07','08','09','10','11','12'],
                        to_path = to_trainpath, 
                        tropical = 'extra',
                        hemi = 'N'
                              )
    print("***********Data extraction complete*************")
=======

    to_trainpath = "datasets/train_N_extra"
    # to_trainpath = "datasets/debugger"

    train_num_subtracks = extract_timeseries(
                                              raw_path = "predictors",
                                              ra_feature_names = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320'],
                                              meta_feature_names = ['pmin', 'x', 'y', 'z', 'month'],
                                              year_range = [2000, 2009], # train_data
                                              months_list = ['01','02','03','04','05','06','07','08','09','10','11','12'],
                                              to_path = to_trainpath, 
                                              tropical = 'extra',
                                              hemi = 'N'
                                                    )
    print("***********Data extraction complete*************")
    train_dataset = Transformer(to_trainpath)
    print("***********Data transformer complete*************")

    train_labels, val_labels, train_meta, val_meta, train_ra, val_ra = train_dataset.train_test_split(split = True)
    print("**********Data split complete*********************")

    pickle.dump(train_labels, open(to_trainpath + "/train_labels.pkl", "wb"))
    pickle.dump(val_labels, open(to_trainpath + "/val_labels.pkl", "wb"))
    pickle.dump(train_meta, open(to_trainpath + "/train_meta.pkl", "wb"))
    pickle.dump(val_meta, open(to_trainpath + "/val_meta.pkl", "wb"))
    pickle.dump(train_ra, open(to_trainpath + "/train_ra.pkl", "wb"))
    pickle.dump(val_ra, open(to_trainpath + "/val_ra.pkl", "wb"))

>>>>>>> 720d9617c54c477a7fedebb76f12daba901f2e07

    # to_testpath = "datasets/test"
    # test_num_subtracks = extract_timeseries(
    #                                         raw_path = "predictors",
    #                                         ra_feature_names = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320'],
    #                                         meta_feature_names = ['pmin', 'x', 'y', 'z', 'month'],
    #                                         year_range = [2009, 2010], # test_data
    #                                         months_list = ['01','02','03','04','05','06','07','08','09','10','11','12'],
    #                                         to_path = "datasets/test", 
    #                                         tropical = 'extra',
    #                                         hemi = 'N'
    #                                     )
    # test_dataset = Transformer("datasets/test")

    # test_labels, test_meta, test_ra = test_dataset.train_test_split(split = False)


    
 
    

    


    



