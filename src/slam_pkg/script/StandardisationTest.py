import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt

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

X = features_df[features]
Y = features_df[target]

x_standardized = scaler_X.transform(X)
y_standardized = scaler_Y.transform(Y)

x_reversed = scaler_X.inverse_transform(x_standardized)
y_reversed = scaler_Y.inverse_transform(y_standardized)

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(10, 25))  # 5 rows, 2 columns

# Plot every original x feature against the reversed x
for i, feature in enumerate(features):
    axs[0, i].scatter(X[feature].values, x_reversed[:, i], alpha=0.5)
    axs[0, i].plot(X[feature].values, X[feature].values, 'r--')  # plot line y=x for reference
    axs[0, i].set_xlabel('Original ' + feature)
    axs[0, i].set_ylabel('Reversed ' + feature)
    axs[0, i].set_title('Original vs Reversed ' + feature)

# For the target comparison plots, plot in the second row
for i, target_variable in enumerate(target):
    column = i % 3  # Ensure we place the plot in the right column
    row = i // 3 + 1  # All target plots go in the second row
    axs[row, column].scatter(Y[target_variable].values, y_reversed[:, i], alpha=0.5)
    axs[row, column].plot(Y[target_variable].values, Y[target_variable].values, 'r--')  # plot line y=x for reference
    axs[row, column].set_xlabel('Original ' + target_variable)
    axs[row, column].set_ylabel('Reversed ' + target_variable)
    axs[row, column].set_title('Original vs Reversed ' + target_variable)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()



