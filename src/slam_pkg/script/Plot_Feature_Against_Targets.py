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

# Create a figure for 3 subplots (3 rows, 1 column)
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot for linear velocity vs delta x
axs[0].scatter(linear_velocity, delta_x, alpha=0.5)
axs[0].set_title('Linear Velocity vs Delta X')
axs[0].set_xlabel('Linear Velocity (m/s)')
axs[0].set_ylabel('Delta X (m)')

# Plot for linear velocity vs delta y
axs[1].scatter(linear_velocity, delta_y, alpha=0.5)
axs[1].set_title('Linear Velocity vs Delta Y')
axs[1].set_xlabel('Linear Velocity (m/s)')
axs[1].set_ylabel('Delta Y (m)')

# Plot for angular velocity vs delta z
axs[2].scatter(angular_velocity, delta_yaw, alpha=0.5)
axs[2].set_title('Angular Velocity vs Delta Z')
axs[2].set_xlabel('Angular Velocity (rad/s)')
axs[2].set_ylabel('Delta Z (rad)')

# Adjust layout to make room for titles and labels
plt.tight_layout()

# Show plot
plt.show()
