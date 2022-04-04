
from dataset import extract_timeseries


if __name__ == '__main__':
    num_subtracks, ra_features, meta_features, labels = extract_timeseries(
                                                                                raw_path = "predictors",
                                                                                ra_feature_names = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320'],
                                                                                meta_feature_names = ['time', 'lon', 'lat', 'pmin', 'id', 'month']
                                                                            )

    print('Number of subtracks: {}'.format(num_subtracks))

    print('Shape of ra_features: {}'.format(ra_features.shape))   

    print('Shape of meta_features: {}'.format(meta_features.shape))  

    print('Shape of labels: {}'.format(labels.shape))



