<<<<<<< HEAD
# dslab2022
 cyclone intensity prediction
=======
# DS Lab 2022 P2 Cyclone

## Description
Predict cyclone intensity and if possible explain feature importance.

## Authors and acknowledgment
Yaqi Qin yaqqin@student.ethz.ch
Feichi Lu feiclu@student.ethz.ch
Tianyang Xu tianyxu@student.ethz.ch

## Current progress
1. RawData class
    (1) load the raw data from the "predictors" folder, and merge them into a dict (add three columns representing X, Y, Z coordinates)
    (2) calculate mean / std of each numeric features in ['lon', 'lat', 'U500', 'V500', 'U300', 'V300', 'T850', 'MSL', 'PV320', 'pmin', 'x', 'y', 'z'] (before calculating, first clip the values, because there are some inf values)
2. extract_time_series function
    (1) current setting: 14 hours as one subtrack, first 7 hours to extract features, last 7 hours to extract labels
    (2) users can specify "tropical" and "hemisphere" to extract
    (3) calculate the moving average in loaddata.py extract_time_series function
3. Transformer class:
    (1) for reanalysis features ['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320']: clip
    (2) for meta features ['pmin', 'x', 'y', 'z', 'month']: replace "month" column with 11-digit one hot encoding ==> ['pmin', 'x', 'y', 'z', 'month-one-hot_1' ~ 'month-one-hot_11']
    (3) train test split: split the preprocessed data based on the assigned ratio 
