import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

dataname = 'Data_Diagonal_Direction.csv'

featurePath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/{dataname}'

features_df = pd.read_csv(featurePath)

# Extract the features and predictions
linear_velocity = features_df['linear_velocity_x'].values
angular_velocity = features_df['angular_velocity_yaw'].values
real_x = features_df['delta_position_x'].values
real_y = features_df['delta_position_y'].values
real_yaw = features_df['delta_yaw'].values

# Assuming delta_t is the time difference between messages (1/30 seconds)
delta_t = 0.0333

# Calculate theta (angle change) from angular velocity
theta = angular_velocity * delta_t

# Calculate linear velocity components
linear_velocity_x_comp = linear_velocity * np.cos(theta)
linear_velocity_y_comp = linear_velocity * np.sin(theta)

# Create a figure for 3 subplots (3 rows, 1 column)
fig, axs = plt.subplots(3, 1, figsize=(10, 15))


# Plot for linear velocity x-component vs delta x and real x
# axs[0].scatter(linear_velocity_x_comp, delta_x, alpha=0.5, color='blue', label='Predicted Delta X')
axs[0].scatter(linear_velocity_x_comp, real_x, alpha=0.5, color='orange', label='Real Delta X')  # Add real Delta X values in orange
axs[0].set_title('Linear Velocity X-Component vs Delta X')
axs[0].set_xlabel('Linear Velocity X-Component (m/s)')
axs[0].set_ylabel('Delta X (m)')
axs[0].legend()

# Plot for linear velocity y-component vs delta y and real y
# axs[1].scatter(linear_velocity_y_comp, delta_y, alpha=0.5, color='blue', label='Predicted Delta Y')
axs[1].scatter(linear_velocity_y_comp, real_y, alpha=0.5, color='orange', label='Real Delta Y')  # Add real Delta Y values in orange
axs[1].set_title('Linear Velocity Y-Component vs Delta Y')
axs[1].set_xlabel('Linear Velocity Y-Component (m/s)')
axs[1].set_ylabel('Delta Y (m)')
axs[1].legend()

# Plot for angular velocity vs delta yaw and real yaw
# axs[2].scatter(angular_velocity, delta_yaw, alpha=0.5, color='blue', label='Predicted Yaw')
axs[2].scatter(angular_velocity, real_yaw, alpha=0.5, color='orange', label='Real Yaw')  # Add real Yaw values in orange
axs[2].set_title('Angular Velocity vs Delta Yaw')
axs[2].set_xlabel('Angular Velocity (rad/s)')
axs[2].set_ylabel('Delta Yaw (rad)')
axs[2].legend()

# Adjust layout to make room for titles and labels
plt.tight_layout()

# Show plot
plt.show()
