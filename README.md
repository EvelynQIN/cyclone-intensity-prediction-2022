<<<<<<< HEAD
# dslab2022
 cyclone intensity prediction
=======
# DS Lab 2022 P2 Cyclone

## Description
Predict cyclone intensity and if possible explain feature importance.

## Authors and acknowledgment
* Yaqi Qin yaqqin@student.ethz.ch
* Feichi Lu feiclu@student.ethz.ch
* Tianyang Xu tianyxu@student.ethz.ch

## Current progress
1. RawData class
    * load the raw data from the "predictors" folder, and merge them into a dict (add three columns representing X, Y, Z coordinates)
    * calculate mean / std of each numeric features (before calculating, first clip the values, because there are some inf values)
2. extract_time_series function
    * current setting: 12 hours as one subtrack, first 6 hours to extract features, last 6 hours to extract labels
    * users can specify "tropical" and "hemisphere" to extract:
        * "tropical": "tropical" :  -20 < lat < 20    vs      "extra": lat < -20 or lat > 20
        * "hemi" : "N" : lat > 0   vs   "S": lat < 0
    *  calculate the moving average in loaddata.py extract_time_series function
3. Transformer class:
    * for **7** reanalysis features **['U300', 'V300', 'U500', 'V500', 'T850', 'MSL', 'PV320']**: clip
    * for meta features ['pmin', 'x', 'y', 'z', 'month']: replace "month" column with 11-digit one hot encoding ==>finally we have **15** meta features **['pmin', 'x', 'y', 'z', 'month-one-hot_1' ~ 'month-one-hot_11']**
    * train test split: split the preprocessed data based on the assigned ratio 
