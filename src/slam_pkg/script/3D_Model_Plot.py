import GPy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pickle
import os

def plot_feature_target(feature_x_index, feature_y_index, target_index, plot_position):
    # Get the feature ranges for the grid
    feature_x_range = np.percentile(X_train[:, feature_x_index], [10, 90])
    feature_y_range = np.percentile(X_train[:, feature_y_index], [10, 90])
    
    # Create the grid
    F1, F2 = np.meshgrid(np.linspace(feature_x_range[0], feature_x_range[1], 50),
                         np.linspace(feature_y_range[0], feature_y_range[1], 50))
    
    # Flatten the grid
    F1_flat = F1.ravel()
    F2_flat = F2.ravel()
    
    # Prepare the full feature matrix
    # Initialize the matrix with repeated means
    other_features_mean = np.mean(X_train, axis=0)
    feature_matrix = np.repeat(other_features_mean.reshape(1, -1), F1_flat.shape[0], axis=0)
    
    # Replace the specific features with grid values
    feature_matrix[:, feature_x_index] = F1_flat
    feature_matrix[:, feature_y_index] = F2_flat
    
    # Predict the deltas for the grid
    delta_predict_mean, delta_predict_variance = loaded_model.predict(feature_matrix)
    
    # Reshape predictions to match the grid
    mean_delta = delta_predict_mean[:, target_index].reshape(F1.shape)
    variance_delta = delta_predict_variance.reshape(F1.shape)
    
    # Plot mean predictions
    ax = fig.add_subplot(3, 2, plot_position, projection='3d')
    surf = ax.plot_surface(F1, F2, mean_delta, cmap='viridis')
    ax.set_xlabel(features[feature_x_index])
    ax.set_ylabel(features[feature_y_index])
    ax.set_zlabel(f'Mean {target[target_index]}')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Plot variance
    ax2 = fig.add_subplot(3, 2, plot_position + 1, projection='3d')
    surf2 = ax2.plot_surface(F1, F2, variance_delta, cmap='hot', alpha=0.7)
    ax2.set_xlabel(features[feature_x_index])
    ax2.set_ylabel(features[feature_y_index])
    ax2.set_zlabel(f'Variance')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # Calculate the Mean Squared Error (MSE) for each feature
    mse_X = np.mean((prediction_data['Predicted_X'] - prediction_data['Real_X']) ** 2)
    mse_Y = np.mean((prediction_data['Predicted_Y'] - prediction_data['Real_Y']) ** 2)
    mse_Yaw = np.mean((prediction_data['Predicted_Yaw'] - prediction_data['Real_Yaw']) ** 2)

    # Overall MSE (considering all features together)
    mse_overall = np.mean((prediction_data[['Predicted_X', 'Predicted_Y', 'Predicted_Yaw']].values - 
                        prediction_data[['Real_X', 'Real_Y', 'Real_Yaw']].values) ** 2)

    plt.figtext(0.5, 0.01, f'Overall MSE: {mse_overall:.4f} | MSE in X: {mse_X:.4f} | MSE in Y: {mse_Y:.4f} | MSE in Yaw: {mse_Yaw:.4f}', 
            ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})



ith_datapoint = 1000
# isSparse = 'sparse0_'
isSparse = ''

modelFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel'
model_filename = f'{isSparse}gpy_model_{ith_datapoint}DP.pkl'

with open(os.path.join(modelFilePath, model_filename), 'rb') as file:
    loaded_model = pickle.load(file)

# Load the standardized predictions and real values
filepath_prediction = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{isSparse}{ith_datapoint}_DP_predictions_vs_real.csv'
filepath_targets = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{isSparse}{ith_datapoint}_DP_train_data.csv'
prediction_data = pd.read_csv(filepath_prediction)
train_data = pd.read_csv(filepath_targets)

features = ['Ground_Truth_X', 'Ground_Truth_Y', 'Ground_Truth_Yaw', 'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw']
target = ['Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']

X_train = train_data[features].values
Y_train = train_data[target].values

fig = plt.figure(figsize=(12, 18))  # Increase figure size to accommodate all subplots

# Indices for plotting in a 3x6 grid (2 for each target, for mean and variance)
plot_idx = 1

feature_pairs = [(0, 3), (1, 4), (2, 5)] 

# Define the number of columns
num_columns = 2

# Iterate over each target
for t_idx, target_name in enumerate(target):
    # Calculate the starting plot index for the current target
    plot_idx = t_idx * num_columns + 1  # Starts at 1, 3, 5 for each row
    
    # Plot the mean for the current target and feature pair
    plot_feature_target(feature_pairs[t_idx][0], feature_pairs[t_idx][1], t_idx, plot_idx)


plt.show()