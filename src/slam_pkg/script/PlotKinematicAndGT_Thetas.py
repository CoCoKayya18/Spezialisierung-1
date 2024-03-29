import pandas as pd
import matplotlib.pyplot as plt

DataFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'

# Load your DataFrame
df = pd.read_csv(DataFilePath)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # 3 rows for x, y, yaw deltas

# Convert Series to NumPy arrays
time_values = df['Time'].values
delta_position_x_values = df['delta_position_x'].values
kinematic_delta_x_values = df['kinematic_delta_x'].values
delta_position_y_values = df['delta_position_y'].values
kinematic_delta_y_values = df['kinematic_delta_y'].values
delta_yaw_values = df['delta_yaw'].values
kinematic_delta_yaw_values = df['kinematic_delta_yaw'].values

# Plot delta_position_x and kinematic_delta_x
axs[0].plot(time_values, delta_position_x_values, label='Delta Position X (Ground Truth)', marker='o', linestyle='-', markersize=2)
axs[0].plot(time_values, kinematic_delta_x_values, label='Kinematic Delta X', marker='x', linestyle='--', markersize=2)
axs[0].set_title('Delta Position X')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Delta X')
axs[0].legend()
axs[0].grid(True)

# Plot delta_position_y and kinematic_delta_y
axs[1].plot(time_values, delta_position_y_values, label='Delta Position Y (Ground Truth)', marker='o', linestyle='-', markersize=2)
axs[1].plot(time_values, kinematic_delta_y_values, label='Kinematic Delta Y', marker='x', linestyle='--', markersize=2)
axs[1].set_title('Delta Position Y')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Delta Y')
axs[1].legend()
axs[1].grid(True)

# Plot delta_yaw and kinematic_delta_yaw
axs[2].plot(time_values, delta_yaw_values, label='Delta Yaw (Ground Truth)', marker='o', linestyle='-', markersize=2)
axs[2].plot(time_values, kinematic_delta_yaw_values, label='Kinematic Delta Yaw', marker='x', linestyle='--', markersize=2)
axs[2].set_title('Delta Yaw')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Delta Yaw (radians)')
axs[2].legend()
axs[2].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()
