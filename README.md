# DS Lab 2022 P2 Cyclone

## Description
This is the 2022 project report of Data Science Lab. The topic of our project is cyclone
intensity prediction. Given meta features and reanalysis features of cyclone tracks from 2000
to 2009 across the globe, we predicted the minimum air pressure at the cyclone center(Pmin)
for the next 6 timesteps with the past 6 timesteps. We conducted different sequence modeling
techniques (LSTM, CNN, TCN, GRU, RNN) on the task, and then focused on feature
importance interpretability using SHAP and model robustness test.

## Model Pipeline
We first pass the reanalysis features through a CNN, then combine the output with the meta features to be the input of the sequence modelling network.
![pipeline](https://github.com/EvelynQIN/cyclone-intensity-prediction-2022/blob/main/imgs/model_pipeline.png "pipeline")

## Results
![MSE](https://github.com/EvelynQIN/cyclone-intensity-prediction-2022/blob/main/imgs/model_comparison.png "MSE")

## Visualization of One Cyclone Track Prediction
![case](https://github.com/EvelynQIN/cyclone-intensity-prediction-2022/blob/main/imgs/vis.png "case")

## Supervisors
* Prof. Dr. Sebastian Schemm sebastian.schemm@env.ethz.ch
* Dr. Michael Armand Sprenger michael.sprenger@env.ethz.ch
* Prof. Dr. Ce Zhang ce.zhang@inf.ethz.ch

## Authors
* Yaqi Qin yaqqin@student.ethz.ch
* Feichi Lu feiclu@student.ethz.ch
* Tianyang Xu tianyxu@student.ethz.ch
