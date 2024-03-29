import pandas as pd
import matplotlib.pyplot as plt

# Load your DataFrame
DataFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/sparse40k_1_DP_predictions_vs_real_test.csv'
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

# Create an index array for the x-axis
index = range(len(df))

# Plot the deltas over time/datapoint index
for i in range(3):
    y_values = [delta_position_x_values, delta_position_y_values, delta_yaw_values][i]
    predicted_values = [predicted_delta_x_values, predicted_delta_y_values, predicted_delta_yaw_values][i]
    axs[i].plot(index, y_values, 'o', label='Ground Truth', markersize=3)
    axs[i].plot(index, predicted_values, 'x', label='Predicted', markersize=3)
    axs[i].set_title(['Delta Position X', 'Delta Position Y', 'Delta Yaw'][i])
    axs[i].set_xlabel('Datapoint Index')
    axs[i].set_ylabel(['Delta X', 'Delta Y', 'Delta Yaw'][i])
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()
