from os import path
import numpy as np

from rawdata import RawData

def extract_timeseries(
    raw_path,
    ra_feature_names,
    meta_feature_names
):
    print("Processing: ", raw_path)

    # Load the data into a RawData object
    data = RawData(raw_path)

    # Get a list of all cyclone tracks
    cyclone_tracks = data.tracks

    num_subtracks = 0

    ra_features = []
    meta_features = []
    labels = []

    # Iterate over each cyclone
    for track in cyclone_tracks:

        # We only want to look at cyclones which live at least 14 hours. (7 h -> pred next 7 h)
        if len(track) < 14:
            continue

        # Extract all possible sub-tracks which contain 14 consecutive time steps 
        sub_tracks = track.extract_all_sub_tracks(1, 14, 1)

        num_subtracks += len(sub_tracks)

        # Iterate over each sub-track
        for sub_track in sub_tracks:

            sub_ra_features = []
            sub_meta_features = []
            sub_labels = []

            # Iterate over the first 7 time steps and extract the data
            for index, step in enumerate(sub_track[:7]):

                # First get the reanalysis features for each of the first 7 time steps
                sub_ra_features.append(step.get_ra_features(ra_feature_names, time_step=index+1).T)

                # Next, get the meta features for each time step
                sub_meta_features.append(step.get_meta_features(meta_feature_names, position_enc="xyz"))

                # Fetch the intensity "pmin" of the time step 7 steps in the future
                label = sub_track[index + 7].get_meta_features(["pmin"])
                sub_labels.append(label)

            ra_features.append([sub_ra_features])
            meta_features.append([sub_meta_features])
            labels.append([sub_labels])  

    # Combine the feature lists
    ra_features = np.vstack(ra_features)
    meta_features = np.vstack(meta_features)
    labels = np.vstack(labels)

    # Reorder the meta features for backwards-compatibility
    """
    pmin = meta_features[:, 0].flatten()
    pcont = meta_features[:, 1].flatten()
    month = meta_features[0, 2]
    month = np.eye(12)[int(month) - 1]
    xyz = meta_features[:, 3:].reshape((21))
    meta_features = np.concatenate((pmin, month, xyz, pcont))  ==>不知道啥意思
    """
    

    # Store all graphs of this file together in one file
    # store_path = path.join(processed_dir, path.basename(raw_path) + ".cube_graphs")
    # torch.save(graph_list, store_path)

    return num_subtracks, ra_features, meta_features, labels
