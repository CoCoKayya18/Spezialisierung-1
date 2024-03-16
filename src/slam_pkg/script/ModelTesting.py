import pickle
import os
import GPy
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def save_predictions_to_csv(predicted_means, real_values, predicted_variances, file_path):
    predicted_means = np.array(predicted_means)
    real_values = np.array(real_values)
    predicted_variances = np.array(predicted_variances)

    # Check if the shapes of the arrays match
    if predicted_means.shape[0] != real_values.shape[0]:
        raise ValueError("The number of predicted means and real values must match.")
    if predicted_variances.shape[0] != predicted_means.shape[0]:
        raise ValueError("The number of predicted variances and predicted means must match.")

    # Create a DataFrame
    columns = ['Predicted_X', 'Predicted_Y', 'Predicted_Yaw', 'Real_X', 'Real_Y', 'Real_Yaw', 'Variance']
    data = np.hstack((predicted_means, real_values, predicted_variances.reshape(-1, 1)))  # Reshape variances for hstack
    df = pandas.DataFrame(data, columns=columns)

    # Save to CSV
    df.to_csv(file_path, index=False)


datafilepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data'
modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'
scalerFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/Scaler'

ith_datapoint = 100

model_filename = f'gpy_model_{ith_datapoint}DP.pkl'
scaler_filenameX = f'scaler_X_{ith_datapoint}.pkl'
scaler_filenameY = f'scaler_Y_{ith_datapoint}.pkl'

# Load the model
with open(os.path.join(modelFilePath, model_filename), 'rb') as file:
    loaded_model = pickle.load(file)

# Get the data out of the csv
dataframe = pandas.read_csv(os.path.join(datafilepath, 'Data.csv'))
features = ['Ground_Truth_X', 'Ground_Truth_Y', 'Ground_Truth_Yaw', 'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw']
target = ['Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']

#get every i-th datapoint out of the csv
ithDataframe = dataframe[::ith_datapoint]
X = ithDataframe[features].values
Y = ithDataframe[target].values

with open(os.path.join(scalerFilePath, scaler_filenameX), 'rb') as file:
    scaler_X = pickle.load(file)

with open(os.path.join(scalerFilePath, scaler_filenameY), 'rb') as file:
    scaler_Y = pickle.load(file)

X_standardized = scaler_X.transform(X)
Y_standardized = scaler_Y.transform(Y)

# Split the data into training, validation and testing sets and save
X_train, X_temp, Y_train, Y_temp = train_test_split(X_standardized, Y_standardized, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Predict using the loaded model
y_predict_mean, y_predict_variance = loaded_model.predict(X_train)

# Print the predictions
print("Predicted Mean:", y_predict_mean)
print("Predicted Variance:", y_predict_variance)

# Print the model parameters
print(loaded_model)

predicted_means_rescaled = scaler_Y.inverse_transform(y_predict_mean)
real_values = scaler_Y.inverse_transform(Y_train)
predicted_variances = y_predict_variance

filename = f'{ith_datapoint}_DP_predictions_vs_real.csv'

file_path = os.path.join(datafilepath, filename)

save_predictions_to_csv(predicted_means_rescaled, real_values, predicted_variances, file_path)
