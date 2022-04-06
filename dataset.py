from os import path
import numpy as np
import pickle

from rawdata import RawData

def extract_timeseries(
    raw_path,
    ra_feature_names,
    meta_feature_names,
    to_path,
    tropical = 'mix',
    hemi = 'mix'
):
    """ for every 14 hour subtracks, extract meta & reanalysis features, labels and moving average
    Args:
        raw_path: the folder path to predictors
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
    data = RawData(raw_path)

    # Get a list of all cyclone tracks
    cyclone_tracks = data.tracks

    num_subtracks = 0

    ra_features = []
    meta_features = []
    labels = []
    moving_avg = []

    # Iterate over each cyclone
    for track in cyclone_tracks:

        # We only want to look at cyclones which live at least 14 hours. (7 h -> pred next 7 h)
        if len(track) < 14:
            continue

        # Extract all possible sub-tracks which contain 14 consecutive time steps 
        sub_tracks, tropical_flag, hemi_flag = track.extract_all_sub_tracks(1, 14, 1)

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

        # Iterate over each sub-track
        for sub_track in sub_tracks:

            sub_ra_features = []
            sub_meta_features = []
            sub_labels = []
            sub_mov_avg = []

            # Iterate over the first 7 time steps and extract the data
            for index, step in enumerate(sub_track[:7]):

                # First get the reanalysis features for each of the first 7 time steps
                sub_ra_features.append(step.get_ra_features(ra_feature_names, time_step=index+1).T)

                # Next, get the meta features for each time step
                sub_meta_features.append(step.get_meta_features(meta_feature_names, position_enc="xyz"))

                # Fetch the intensity "pmin" of the time step 7 steps in the future
                label = sub_track[index + 7].get_meta_features(["pmin"])[0]
                sub_labels.append(label)

                # compute the moving average as the baseline
                pmin_avg = np.mean([t.get_meta_features(["pmin"]) for t in sub_track[index: index + 7]])
                sub_mov_avg.append(pmin_avg)


            ra_features.append([sub_ra_features])
            meta_features.append([sub_meta_features])
            labels.append([sub_labels]) 
            moving_avg.append([sub_mov_avg])

            

    # Combine the feature lists
    ra_features = np.vstack(ra_features)
    meta_features = np.vstack(meta_features)
    labels = np.vstack(labels)
    moving_avg = np.vstack(moving_avg)

    # save data to pkl files
    pickle.dump(ra_features, open(to_path + "/ra_features.pkl", "wb"))
    pickle.dump(meta_features, open(to_path + "/meta_features.pkl", "wb"))
    pickle.dump(labels, open(to_path + "/labels.pkl", "wb"))
    pickle.dump(moving_avg, open(to_path + "/moving_avg.pkl", "wb"))


    return num_subtracks, ra_features, meta_features, labels, moving_avg
