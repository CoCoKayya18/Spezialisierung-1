import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'predictions_vs_real.csv' is the file that contains the prediction and real data
DataFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/sparse0_1_DP_predictions_vs_real_test.csv'

# Load your DataFrame
df = pd.read_csv(DataFilePath)

limit = False

df = pd.read_csv(DataFilePath)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # 3 rows for delta x, y, yaw

# Extracting the values
delta_position_x_values = df['Real_X'].values
predicted_delta_x_values = df['Predicted_X'].values
delta_position_y_values = df['Real_Y'].values
predicted_delta_y_values = df['Predicted_Y'].values
delta_yaw_values = df['Real_Yaw'].values
predicted_delta_yaw_values = df['Predicted_Yaw'].values

# Set specific limits for x and y axes based on the data
x_limits = [(-0.05, 0.05), (-0.05, 0.05), (-0.2, 0.2)]
y_limits = [(-0.05, 0.05), (-0.05, 0.05), (-0.2, 0.2)]


for i in range(3):
    axs[i].plot(delta_position_x_values, predicted_delta_x_values, 'o', label='Predicted vs Real')
    axs[i].set_title(['Delta Position X', 'Delta Position Y', 'Delta Yaw'][i])
    axs[i].set_xlabel(['Real X', 'Real Y', 'Real Yaw'][i])
    axs[i].set_ylabel(['Predicted X', 'Predicted Y', 'Predicted Yaw'][i])
    axs[i].legend()
    axs[i].grid(True)
    if limit:
        axs[i].set_xlim(x_limits[i])
        axs[i].set_ylim(y_limits[i])

plt.tight_layout()
plt.show()
