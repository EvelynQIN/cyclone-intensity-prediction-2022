from os import path
import numpy as np
import pandas as pd
import pickle

from rawdata import RawData

def extract_timeseries(
    raw_path,
    ra_feature_names,
    meta_feature_names,
    to_path,
    start_stride, 
    split, 
    split_ratio,
    year_range,
    months_list = ['01','02','03','04','05','06','07','08','09','10','11','12'], 
    tropical = 'mix',
    hemi = 'mix'
):
    """ for every 14 hour subtracks, extract meta & reanalysis features, labels and moving average
    Args:
        raw_path: the folder path to predictors
        start_stride:the stride of the first element of a subtrack
        split: boolean as to whether to split the dataset into train and val
        split_ratio: [0, 1]
        ra_feature_names: subset of ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320']
        meta_feature_names: subset of ['time', 'lon', 'lat', 'pmin', 'id', 'month']
        tropical = ['tropical', 'extra', 'mix'], default = 'mix'
        hemi = ['N', 'S', 'mix'], default = 'mix'
    Returns:
        num_subtracks
        ra_features
        meta_features
        labels
        moving_avg
    """
    print("Processing: ", raw_path)

    # Load the data into a RawData object
    data = RawData(raw_path, year_range, months_list)
    #data = pd.read_pickle(to_path + "/raw.pkl")

    pickle.dump(data._dataset['id'], open(to_path + "/cyclone_ids.pkl", "wb"))  # save the raw data into pickle
    #pickle.dump(data, open(to_path + "/raw.pkl", "wb"))

    print("===========extract_timeseries.Raw data complete!============")

    # Get the scaler dict
    scaler_dict = data.mean_std_cal(tropical, hemi)
    print("===========extract_timeseries.mean_std_cal complete!============")

    # Get a list of all cyclone tracks
    cyclone_tracks = data.tracks

    num_subtracks = 0

    ra_features = []
    meta_features = []
    labels = []
    # ra_features = np.array([])
    # meta_features = np.array([])
    # labels = np.array([])
    # moving_avg = []

    # Iterate over each cyclone
    for (track_id , track) in enumerate(cyclone_tracks):

        # We only want to look at cyclones which live at least 14 hours. (7 h -> pred next 7 h)
        if len(track) < 12:
            continue

        # Extract all possible sub-tracks which contain 12 consecutive time steps 
        sub_tracks, tropical_flag, hemi_flag = track.extract_all_sub_tracks(start_stride, 12, 1)

        if tropical == 'tropical' and hemi == 'N':
            index = np.intersect1d(np.where(tropical_flag == 1), np.where(hemi_flag == 1))
            sub_tracks = sub_tracks[index]
        elif tropical == 'extra' and hemi == 'N':
            index = np.intersect1d(np.where(tropical_flag == 0), np.where(hemi_flag == 1))
            sub_tracks = sub_tracks[index]
        elif tropical == 'tropical' and hemi == 'S':
            index = np.intersect1d(np.where(tropical_flag == 1), np.where(hemi_flag == 0))
            sub_tracks = sub_tracks[index]
        elif tropical == 'extra' and hemi == 'S':
            index = np.intersect1d(np.where(tropical_flag == 0), np.where(hemi_flag == 0))
            sub_tracks = sub_tracks[index]
        elif tropical == 'tropical' and hemi == 'mix':
            index = np.where(tropical_flag == 1)[0]
            sub_tracks = sub_tracks[index]
        elif tropical == 'extra' and hemi == 'mix':
            index = np.where(tropical_flag == 0)[0]
            sub_tracks = sub_tracks[index]
        elif tropical == 'mix' and hemi == 'N':
            index = np.where(hemi_flag == 1)[0]
            sub_tracks = sub_tracks[index]
        elif tropical == 'mix' and hemi == 'S':
            index = np.where(hemi_flag == 0)[0]
            sub_tracks = sub_tracks[index]           

        num_subtracks += len(sub_tracks)

        # print num of subtracks info
        if track_id % 10000 == 0:
            print("extracting {} cyclones with {} subtracks".format(track_id, num_subtracks))

        # Iterate over each sub-track
        for sub_track in sub_tracks:

            sub_ra_features = []
            sub_meta_features = []
            sub_labels = []
            # sub_mov_avg = []

            # Iterate over the first 7 time steps and extract the data
            for index, step in enumerate(sub_track[:6]):

                # First get the reanalysis features for each of the first 7 time steps
                sub_ra_features.append(step.get_ra_features(ra_feature_names, time_step=index+1).T)

                # Next, get the meta features for each time step
                sub_meta_features.append(step.get_meta_features(meta_feature_names))

                # Fetch the intensity "pmin" of the time step 7 steps in the future
                label = sub_track[index + 6].get_meta_features(["pmin"])[0]
                sub_labels.append(label)
                # sub_labels = np.array([label]) if sub_labels.shape[0] == 0 else np.append(sub_labels, [label], axis = 0)

                # # compute the moving average as the baseline
                # curr_pmins = [t.get_meta_features(["pmin"]) for t in sub_track[index: 6]]
                # curr_pmins = np.append(curr_pmins, sub_mov_avg)
                # pmin_avg = np.mean(curr_pmins)
                # sub_mov_avg.append(pmin_avg)


            ra_features.append(sub_ra_features) 
            meta_features.append(sub_meta_features)
            labels.append(sub_labels)
            # moving_avg.append([sub_mov_avg])

    print("===========extract_timeseries: {} cyclone tracks with {} sub-tracks extraction complete!============".format(track_id, num_subtracks))    

    # Combine the feature lists
    # ra_features = np.vstack(ra_features)
    # meta_features = np.vstack(meta_features)
    # labels = np.vstack(labels)
    # moving_avg = np.vstack(moving_avg)
    ra_features = np.array(ra_features)
    meta_features = np.array(meta_features)
    labels = np.array(labels)

    if split:
        print("Start train val split!")
        datasize = labels.shape[0]
        split_ind = int(datasize * split_ratio)
        print("train size: {}   ||    val size {}".format(split_ind, datasize - split_ind))

        print('Shape of train_meta_features: {}'.format(meta_features[:split_ind].shape))  

        print('Shape of val_meta_features: {}'.format(meta_features[split_ind:].shape))  

        print('Shape of train_ra_features: {}'.format(ra_features[:split_ind].shape))  

        print('Shape of val_ra_features: {}'.format(ra_features[split_ind:].shape))  
        
        print('Shape of train_labels: {}'.format(labels[:split_ind].shape))  

        print('Shape of val_labels: {}'.format(labels[split_ind:].shape))

        pickle.dump(labels[:split_ind], open(to_path + "/train_labels.pkl", "wb"), protocol = 4)
        pickle.dump(labels[split_ind:], open(to_path+ "/val_labels.pkl", "wb"), protocol = 4)
        pickle.dump(meta_features[:split_ind], open(to_path + "/train_meta.pkl", "wb"), protocol = 4)
        pickle.dump(meta_features[split_ind:], open(to_path + "/val_meta.pkl", "wb"), protocol = 4)
        pickle.dump(ra_features[:split_ind], open(to_path + "/train_ra.pkl", "wb"), protocol = 4)
        pickle.dump(ra_features[split_ind:], open(to_path+ "/val_ra.pkl", "wb"), protocol = 4)
        pickle.dump(scaler_dict, open(to_path + "/scaler_dict.pkl", "wb"))
        print("**********Data split and save complete*********************")  
        print("======================================================")
    else:
        print("Start test data processing!")

        print('Shape of test_meta_features: {}'.format(meta_features.shape))   

        print('Shape of test_ra_features: {}'.format(ra_features.shape))  
              
        print('Shape of test_labels: {}'.format(labels.shape))  

        # save data to pkl files
        pickle.dump(ra_features, open(to_path + "/ra_features.pkl", "wb"), protocol = 4)
        pickle.dump(meta_features, open(to_path + "/meta_features.pkl", "wb"), protocol = 4)
        pickle.dump(labels, open(to_path + "/labels.pkl", "wb"), protocol = 4)
        # pickle.dump(moving_avg, open(to_path + "/moving_avg.pkl", "wb"))
        pickle.dump(scaler_dict, open(to_path + "/scaler_dict.pkl", "wb"))
        print("===========extract_timeseries.test data save complete!============") 
        



