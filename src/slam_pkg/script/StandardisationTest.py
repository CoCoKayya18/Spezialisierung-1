import pickle
import pandas as pd
import os

ith_datapoint = 1
isSparse = 'sparseKFold1_'
# isSparse = ''
# isTuned = 'BayesianOptimizationTuned_'
# isTuned = isTuned + 'GridSearchTuned_'
# isTuned = 'BayesianOptimizationTuned_GridSearchTuned_'
isTuned = ''
# SpecialCase = '_OneDirection'
SpecialCase = ''
dataname = 'Data_Only_X_Direction.csv'

featurePath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{dataname}'
scalerFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/Scaler'

scaler_filenameX = f'{isSparse}scaler_X_{ith_datapoint}{SpecialCase}.pkl'
scaler_filenameY = f'{isSparse}scaler_Y_{ith_datapoint}{SpecialCase}.pkl'

with open(os.path.join(scalerFilePath, scaler_filenameX), 'rb') as file:
    scaler_X = pickle.load(file)

with open(os.path.join(scalerFilePath, scaler_filenameY), 'rb') as file:
    scaler_Y = pickle.load(file)

features_df = pd.read_csv(featurePath)

features = ['linear_velocity_x', 'angular_velocity_yaw']
target = ['delta_position_x', 'delta_position_y', 'delta_yaw']

standardized_df_x = features_df[features]
standardized_df_y = features_df[target]

