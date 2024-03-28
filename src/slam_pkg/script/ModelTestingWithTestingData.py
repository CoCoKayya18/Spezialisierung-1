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
isSparse = 'sparse0_'
# isSparse = ''
# isTuned = 'BayesianOptimizationTuned_'
# isTuned = isTuned + 'GridSearchTuned_'
# isTuned = 'BayesianOptimizationTuned_GridSearchTuned_'
isTuned = ''

model_filename = f'{isTuned}{isSparse}gpy_model_{ith_datapoint}DP.pkl'
scaler_filenameX = f'{isSparse}scaler_X_{ith_datapoint}.pkl'
scaler_filenameY = f'{isSparse}scaler_Y_{ith_datapoint}.pkl'

# Load the model

if isTuned == '':
    with open(os.path.join(modelFilePath, model_filename), 'rb') as file:
        loaded_model = pickle.load(file)

if isTuned != '':
    with open(os.path.join(tunedModelFilePath, model_filename), 'rb') as file:
        loaded_model = pickle.load(file)

# Get the data out of the csv
dataframe = pandas.read_csv(os.path.join(datafilepath, f'{isSparse}{ith_datapoint}_DP_train_data.csv'))

if isTuned == '':
    validationDataFrame = pandas.read_csv(os.path.join(datafilepath, f'{isSparse}{ith_datapoint}_DP_val_data.csv'))
    dataframe = pandas.concat([dataframe, validationDataFrame], ignore_index=True)

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

# Predict using the loaded model
y_predict_mean, y_predict_variance = loaded_model.predict(X_train)

# Print the model parameters
print(loaded_model)

predicted_means_rescaled = scaler_Y.inverse_transform(y_predict_mean)
real_values = scaler_Y.inverse_transform(Y_train)
predicted_variances = y_predict_variance

filename = f'{isTuned}{isSparse}{ith_datapoint}_DP_predictions_vs_real_test.csv'

file_path = os.path.join(datafilepath, filename)


save_predictions_to_csv(predicted_means_rescaled, real_values, kinematic_values, file_path)

# save_predictions_to_csv(predicted_means_rescaled, real_values, predicted_variances, file_path)

# Plot Model

# variance_scaling_factors = scaler_Y.scale_ ** 2  # Assuming all targets share the same scale factor
# scaled_variances = y_predict_variance.flatten() * variance_scaling_factors[0]  # Use the first scale factor as an example

# # Calculate confidence intervals for the predictions
# confidence_upper = predicted_means_rescaled + 1.96 * np.sqrt(scaled_variances)[:, None]  # Add new axis for broadcasting
# confidence_lower = predicted_means_rescaled - 1.96 * np.sqrt(scaled_variances)[:, None]

# # For plotting, let's take one feature (feature_index) from X_train to plot against
# # feature_index = 0  # Index for the first feature
# n_features = X_train.shape[1]
# n_targets = Y_train.shape[1]

# feature_target_pairs = [(0, 0), (3, 0), (1, 1), (4, 1), (2, 2), (5, 2)]

# # Create a figure and a set of subplots
# n_plots = len(feature_target_pairs)
# plt.figure(figsize=(12, 6 * n_plots))

# for i, (feature_index, target_index) in enumerate(feature_target_pairs, start=1):

#     x_values_for_plotting = X_train[:, feature_index]  # Get the values of the feature for all samples

#     # Sort the X values for plotting (and sort everything else accordingly)
#     sorted_indices = np.argsort(x_values_for_plotting)  # Get sorted indices of X
#     x_values_sorted = x_values_for_plotting[sorted_indices]
#     predicted_means_sorted = predicted_means_rescaled[sorted_indices]
#     confidence_upper_sorted = confidence_upper[sorted_indices]
#     confidence_lower_sorted = confidence_lower[sorted_indices]

#     ax = plt.subplot(n_plots, 1, i)

#     # Plot the mean prediction and actual values
#     ax.plot(x_values_sorted, predicted_means_sorted[:, target_index], 'b-', label='Mean Prediction')
#     ax.plot(x_values_sorted, real_values[sorted_indices, target_index], 'rx', label='Actual Values')

#     # Plot the confidence interval
#     ax.fill_between(x_values_sorted, confidence_lower_sorted[:, target_index], confidence_upper_sorted[:, target_index], alpha=0.3, label='95% Confidence Interval')

#     feature_name = features[feature_index] if feature_index < len(features) else f"Feature {feature_index}"
#     target_name = target[target_index] if target_index < len(target) else f"Target {target_index}"

#     ax.set_xlabel(f'Real {feature_name}')
#     ax.set_ylabel(f'Predicted/Real {target_name}')
#     ax.set_title(f'Gaussian Process Regression Predictions: {feature_name} vs. {target_name}')
#     ax.legend()

# plt.legend()
# plt.show()
