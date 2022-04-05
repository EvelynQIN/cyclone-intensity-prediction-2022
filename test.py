
from dataset import extract_timeseries
from moving_average import moving_average
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    num_subtracks, ra_features, meta_features, labels, moving_avg = extract_timeseries(
                                                                                raw_path = "predictors",
                                                                                ra_feature_names = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320'],
                                                                                meta_feature_names = ['time', 'lon', 'lat', 'pmin', 'id', 'month'],
                                                                                tropical = 'tropical',
                                                                                hemi = 'mix'

                                                                            )

    print('Number of subtracks: {}'.format(num_subtracks))

    print('Shape of ra_features: {}'.format(ra_features.shape))   

    print('Shape of meta_features: {}'.format(meta_features.shape))  

    print('Shape of labels: {}'.format(labels.shape))

    print('Shape of moving_avg: {}'.format(moving_avg.shape))

    pred_intensity = moving_average(meta_features)
    true_intensity = labels.reshape(-1, labels.shape[1])
    mean_squared_error(pred_intensity, true_intensity)



