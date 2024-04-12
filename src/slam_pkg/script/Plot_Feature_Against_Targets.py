import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ith_datapoint = 1
# isSparse = 'sparse40k_'
isSparse = ''
# isTuned = 'BayesianOptimizationTuned_'
# isTuned = 'GridSearchTuned_'
# isTuned = 'BayesianOptimizationTuned_GridSearchTuned_'
isTuned = ''
trainOrTest = '_test'
# trainOrTest = '_train'
SpecialCase = '_OneDirection'
# SpecialCase = ''

featurePathTest = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{isTuned}{isSparse}{ith_datapoint}_DP{trainOrTest}_data{SpecialCase}.csv'
featurePathVal = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{isTuned}{isSparse}{ith_datapoint}_DP{trainOrTest}_data{SpecialCase}.csv'
targetPath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{isTuned}{isSparse}{ith_datapoint}{SpecialCase}_DP_predictions_vs_real{trainOrTest}.csv'

# Load the data
if trainOrTest == '_test':
    featuresTest_df = pd.read_csv(featurePathTest)
    featureVal_df = pd.read_csv(featurePathVal)
    
    features_df = pd.concat([featuresTest_df, featureVal_df], ignore_index=True)

if trainOrTest == '_train':
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

# Create a figure for 3 subplots (3 rows, 1 column)
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot for linear velocity vs delta x and real x
axs[0].scatter(linear_velocity, delta_x, alpha=0.5, color='blue', label='Predicted X')
axs[0].scatter(linear_velocity, real_x, alpha=0.5, color='orange', label='Real X')  # Add real X values in orange
axs[0].set_title('Linear Velocity vs Delta X')
axs[0].set_xlabel('Linear Velocity (m/s)')
axs[0].set_ylabel('Delta X (m)')
axs[0].legend()

# Plot for linear velocity vs delta y and real y
axs[1].scatter(linear_velocity, delta_y, alpha=0.5, color='blue', label='Predicted Y')
axs[1].scatter(linear_velocity, real_y, alpha=0.5, color='orange', label='Real Y')  # Add real Y values in orange
axs[1].set_title('Linear Velocity vs Delta Y')
axs[1].set_xlabel('Linear Velocity (m/s)')
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
