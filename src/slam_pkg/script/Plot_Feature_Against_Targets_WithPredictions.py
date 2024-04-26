import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

ith_datapoint = 1
isSparse = 'sparseKFoldSquareRobotFrameDirection_'
# isSparse = ''
SpecialCase = '_Square_RobotFrameDeltas_Direction'
# SpecialCase = ''
# dataName = 'Data.csv'
dataName = 'Data_Square_RobotFrameDeltas_Direction'
# isTuned = 'BayesianOptimizationTuned_'
# isTuned = 'GridSearchTuned_'
# isTuned = 'BayesianOptimizationTuned_GridSearchTuned_'
isTuned = ''
# trainOrTest = '_train'
trainOrTest = '_test'

featurePathTest = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{isTuned}{isSparse}{ith_datapoint}_DP{trainOrTest}_data{SpecialCase}.csv'
featurePathVal = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{isTuned}{isSparse}{ith_datapoint}_DP{trainOrTest}_data{SpecialCase}.csv'
targetPath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{isTuned}{isSparse}{ith_datapoint}{SpecialCase}_DP_predictions_vs_real{trainOrTest}.csv'

# Load the data
# if trainOrTest == '_test':
#     featuresTest_df = pd.read_csv(featurePathTest)
#     featureVal_df = pd.read_csv(featurePathVal)
    
#     features_df = pd.concat([featuresTest_df, featureVal_df], ignore_index=True)

# if trainOrTest == '_train':

features_df = pd.read_csv(featurePathTest)

predictions_df = pd.read_csv(targetPath)

min_length = min(len(features_df), len(predictions_df))
predictions_df = predictions_df.head(min_length)
features_df = features_df.head(min_length)

# Check if the features and targets have the same number of rows
if len(features_df) != len(predictions_df):
    print(len(features_df)) 
    print(len(predictions_df))
    raise ValueError("The number of rows in features and predictions does not match.")

# Extract the features and predictions
linear_velocity = features_df['linear_velocity_x'].values
angular_velocity = features_df['angular_velocity_yaw'].values
delta_x = predictions_df['Predicted_X'].values
delta_y = predictions_df['Predicted_Y'].values
delta_yaw = predictions_df['Predicted_Yaw'].values  # Assuming Predicted_Yaw is delta_z
real_x = predictions_df['Real_X'].values
real_y = predictions_df['Real_Y'].values
real_yaw = predictions_df['Real_Yaw'].values

# Assuming delta_t is the time difference between messages (1/30 seconds)
delta_t = 0.0333

# Calculate theta (angle change) from angular velocity
theta = angular_velocity * delta_t

# Calculate linear velocity components
linear_velocity_x_comp = linear_velocity * np.cos(theta)
linear_velocity_y_comp = linear_velocity * np.sin(theta)

# Create a figure for 3 subplots (3 rows, 1 column)
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# # Plot for linear velocity vs delta x and real x
# axs[0].scatter(linear_velocity, delta_x, alpha=0.5, color='blue', label='Predicted X')
# axs[0].scatter(linear_velocity, real_x, alpha=0.5, color='orange', label='Real X')  # Add real X values in orange
# axs[0].set_title('Linear Velocity vs Delta X')
# axs[0].set_xlabel('Linear Velocity (m/s)')
# axs[0].set_ylabel('Delta X (m)')
# axs[0].legend()

# # Plot for linear velocity vs delta y and real y
# axs[1].scatter(linear_velocity, delta_y, alpha=0.5, color='blue', label='Predicted Y')
# axs[1].scatter(linear_velocity, real_y, alpha=0.5, color='orange', label='Real Y')  # Add real Y values in orange
# axs[1].set_title('Linear Velocity vs Delta Y')
# axs[1].set_xlabel('Linear Velocity (m/s)')
# axs[1].set_ylabel('Delta Y (m)')
# axs[1].legend()

# Plot for linear velocity x-component vs delta x and real x
axs[0].scatter(linear_velocity_x_comp, delta_x, alpha=0.5, color='blue', label='Predicted Delta X')
axs[0].scatter(linear_velocity_x_comp, real_x, alpha=0.5, color='orange', label='Real Delta X')  # Add real Delta X values in orange
axs[0].set_title('Linear Velocity X-Component vs Delta X')
axs[0].set_xlabel('Linear Velocity X-Component (m/s)')
axs[0].set_ylabel('Delta X (m)')
axs[0].legend()

# Plot for linear velocity y-component vs delta y and real y
axs[1].scatter(linear_velocity_y_comp, delta_y, alpha=0.5, color='blue', label='Predicted Delta Y')
axs[1].scatter(linear_velocity_y_comp, real_y, alpha=0.5, color='orange', label='Real Delta Y')  # Add real Delta Y values in orange
axs[1].set_title('Linear Velocity Y-Component vs Delta Y')
axs[1].set_xlabel('Linear Velocity Y-Component (m/s)')
axs[1].set_ylabel('Delta Y (m)')
axs[1].legend()

# Plot for angular velocity vs delta yaw and real yaw
axs[2].scatter(angular_velocity, delta_yaw, alpha=0.5, color='blue', label='Predicted Yaw')
axs[2].scatter(angular_velocity, real_yaw, alpha=0.5, color='orange', label='Real Yaw')  # Add real Yaw values in orange
axs[2].set_title('Angular Velocity vs Delta Yaw')
axs[2].set_xlabel('Angular Velocity (rad/s)')
axs[2].set_ylabel('Delta Yaw (rad)')
axs[2].legend()

# Adjust layout to make room for titles and labels
plt.tight_layout()

# Show plot
plt.show()
