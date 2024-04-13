import pickle
import os
import GPy
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def save_predictions_to_csv(predicted_means, real_values, kinematic_deltas, file_path):
    predicted_means = np.array(predicted_means)
    real_values = np.array(real_values)
    kinematic_deltas = np.array(kinematic_deltas)

    # Check if the shapes of the arrays match
    if predicted_means.shape[0] != real_values.shape[0]:
        raise ValueError("The number of predicted means and real values must match.")
    if predicted_variances.shape[0] != predicted_means.shape[0]:
        raise ValueError("The number of predicted variances and predicted means must match.")
    if isinstance(kinematic_deltas, pandas.DataFrame):
        kinematic_deltas = kinematic_deltas[['kinematic_delta_x', 'kinematic_delta_y', 'kinematic_delta_yaw']].values

    combined_data = np.hstack((predicted_means, real_values, kinematic_deltas))  # Reshape variances for hstack

    # Create a DataFrame
    columns = ['Predicted_X', 'Predicted_Y', 'Predicted_Yaw', 'Real_X', 'Real_Y', 'Real_Yaw', 'Kinematic_X', 'Kinematic_Y', 'Kinematic_Yaw']

    final_df = pandas.DataFrame(combined_data, columns=columns)
    
    # Calculate kinematic deltas and add them to the DataFrame
    # predictions_df = calculate_kinematic_deltas(predictions_df)
    
    # Save the DataFrame to CSV
    final_df.to_csv(file_path, index=False)


datafilepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data'
modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'
tunedModelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodelTuned'
scalerFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/Scaler'

ith_datapoint = 1
isSparse = 'sparseKFoldDiagonalFirstDirection_'
# isSparse = ''
SpecialCase = '_One_Full_Diagonal_Direction'
# SpecialCase = ''
# dataName = 'Data.csv'
dataFile = 'Data_Third_Diagonal_Direction.csv'
dataName = 'Data_Third_Diagonal_Direction'
# isTuned = 'BayesianOptimizationTuned_'
# isTuned = isTuned + 'GridSearchTuned_'
# isTuned = 'BayesianOptimizationTuned_GridSearchTuned_'
isTuned = ''

model_filename = f'{isTuned}{isSparse}gpy_model_{ith_datapoint}DP{SpecialCase}.pkl'
scaler_filenameX = f'{isSparse}scaler_X_{ith_datapoint}{SpecialCase}.pkl'
scaler_filenameY = f'{isSparse}scaler_Y_{ith_datapoint}{SpecialCase}.pkl'

# Load the model

if isTuned == '':
    with open(os.path.join(modelFilePath, model_filename), 'rb') as file:
        loaded_model = pickle.load(file)

if isTuned != '':
    with open(os.path.join(tunedModelFilePath, model_filename), 'rb') as file:
        loaded_model = pickle.load(file)

# Get the data out of the csv
dataframe = pandas.read_csv(os.path.join(datafilepath, dataFile))

# if isTuned == '':
#     validationDataFrame = pandas.read_csv(os.path.join(datafilepath, f'{isSparse}{ith_datapoint}_DP_val_data{SpecialCase}.csv'))
#     dataframe = pandas.concat([dataframe, validationDataFrame], ignore_index=True)

features = ['linear_velocity_x', 'angular_velocity_yaw']
target = ['delta_position_x', 'delta_position_y', 'delta_yaw']
kinematicDeltas = ['kinematic_delta_x', 'kinematic_delta_y', 'kinematic_delta_yaw']

#get every i-th datapoint out of the csv
# ithDataframe = dataframe[::ith_datapoint]
X_train = dataframe[features].values
Y_train = dataframe[target].values
kinematic_values = dataframe[kinematicDeltas].values

with open(os.path.join(scalerFilePath, scaler_filenameX), 'rb') as file:
    scaler_X = pickle.load(file)

with open(os.path.join(scalerFilePath, scaler_filenameY), 'rb') as file:
    scaler_Y = pickle.load(file)

X_train_standardized = scaler_X.transform(X_train)

# Predict using the loaded model and the standardized data
y_predict_mean, y_predict_variance = loaded_model.predict(X_train_standardized)

# Print the model parameters
print(loaded_model)

predicted_means_rescaled = scaler_Y.inverse_transform(y_predict_mean)
real_values = Y_train
predicted_variances = y_predict_variance

filename = f'{isTuned}{isSparse}{ith_datapoint}{dataName}_DP_predictions_vs_real_new.csv'

file_path = os.path.join(datafilepath, filename)


save_predictions_to_csv(predicted_means_rescaled, real_values, kinematic_values, file_path)
