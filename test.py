
from dataset import extract_timeseries
from loaddata import Transformer


if __name__ == '__main__':
    num_subtracks, ra_features, meta_features, labels, moving_avg = extract_timeseries(
                                                                                raw_path = "predictors",
                                                                                ra_feature_names = ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320'],
                                                                                meta_feature_names = ['time', 'lon', 'lat', 'pmin', 'id', 'month'], # ['time', 'pmin', 'id', 'month', 'x', 'y', 'z']
                                                                                to_path = "datasets",
                                                                                tropical = 'tropical',
                                                                                hemi = 'S'
                                                                            )
    dataset = Transformer("datasets")

    train_labels, test_labels, train_meta, test_meta, train_ra, test_ra = dataset.train_test_split()
    
    print('Number of subtracks: {}'.format(num_subtracks))

    print('Shape of ra_features: {}'.format(ra_features.shape))   

    print('Shape of meta_features: {}'.format(meta_features.shape))  

    print('Shape of labels: {}'.format(labels.shape))

    print('Shape of moving_avg: {}'.format(moving_avg.shape))

    print('Shape of train_meta_features: {}'.format(train_meta.shape))  

    print('Shape of test_meta_features: {}'.format(test_meta.shape))  

    print('Shape of train_ra_features: {}'.format(train_ra.shape))  

    print('Shape of test_ra_features: {}'.format(test_ra.shape))  
    
    print('Shape of train_labels: {}'.format(train_labels.shape))  

    print('Shape of test_labels: {}'.format(test_labels.shape))  

    



