import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the complete filtered_dfset
filtered_dfframe = pd.read_csv('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data_Square_World_Direction.csv')

# List all columns that you want to keep. Adjust this list based on your specific filtered_dfset and the columns you are interested in.
columns_of_interest = [
    'Time',
    'linear_velocity_x', 'angular_velocity_yaw',  # These might be computed or derived values
    'twist.twist.linear.x', 'twist.twist.angular.z',  # Example odometry velocities
    'linear.x', 'angular.z'  # Example commanded velocities
]

time_diff = 0.033

# Filter out the filtered_dfframe to only include the columns of interest
filtered_df = filtered_dfframe[columns_of_interest]

time_values = filtered_df['Time'].values

calculated_linear = filtered_df['linear_velocity_x'].values
# calculated_world_y = filtered_df['world_velocity_y'].values
odometry_linear = filtered_df['twist.twist.linear.x'].values
calculated_angular = filtered_df['angular_velocity_yaw'].values
odometry_angular = filtered_df['twist.twist.angular.z'].values

# theta = 3.1415 + np.cumsum(np.insert(filtered_df['angular_velocity_yaw'].values, 0, 0)[:-1]) * time_diff
# theta = (theta + np.pi) % (2 * np.pi) - np.pi
# theta = np.mod(theta, 2 * np.pi)

# print(theta)
# theta_df = pd.DataFrame({
#     'Time': time_values,
#     'Theta': theta
# })
# theta_df.to_csv('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/odom_thetas.csv')

# odometry_world_x = odometry_linear * np.cos(theta)
# odometry_world_y = odometry_linear * np.sin(theta)

# Calculate Bland-Altman plot statistics
def bland_altman_stats(data1, data2):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff, axis=0)
    sd = np.std(diff, axis=0)
    return mean, diff, md, sd

# Create a Bland-Altman plot
def bland_altman_plot(mean, diff, md, sd, title, subplot_index):
    plt.subplot(3, 2, subplot_index)
    plt.scatter(mean, diff, alpha=0.5, s=1)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--', linewidth=2)
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--', linewidth=2)
    plt.title(title)
    plt.xlabel('Mean Value (m/s or rad/s)')
    plt.ylabel('Difference (m/s or rad/s)')

# Error analysis and plot function
def error_analysis_and_plot(errors, title, subplot_index):
    mse = np.mean(np.square(errors))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    print(f"{title} - Mean Squared Error: {mse}, Root Mean Squared Error: {rmse}, Mean Absolute Error: {mae}")

    plt.subplot(3, 2, subplot_index)
    plt.hist(errors, bins=50, alpha=0.5)
    plt.title(f'{title} - Error Distribution')
    plt.xlabel('Errors')
    plt.ylabel('Frequency')

    # Display MSE, RMSE, and MAE on the plot
    error_text = f'MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}'
    plt.text(0.05, 0.95, error_text, transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Create figure for plotting
plt.figure(figsize=(15, 15))

# Plot linear velocities
plt.subplot(3, 2, 1)
# plt.plot(filtered_df['Time'].values, calculated_linear, label='Calculated Linear Velocity', marker='.')
# plt.plot(filtered_df['Time'].values, odometry_linear, label='Odometry Linear Velocity', marker='.')
# plt.title('Comparison of Linear Velocities')
# plt.xlabel('Time (s)')
# plt.ylabel('Linear Velocity (m/s)')
# plt.legend()
# plt.grid(True)

# # Plot angular velocities
# plt.subplot(3, 2, 2)
# plt.plot(filtered_df['Time'].values, calculated_angular, label='Calculated Angular Velocity', marker='.')
# plt.plot(filtered_df['Time'].values, odometry_angular, label='Odometry Angular Velocity', marker='.')
# plt.title('Comparison of Angular Velocities')
# plt.xlabel('Time (s)')
# plt.ylabel('Angular Velocity (rad/s)')
# plt.legend()
# plt.grid(True)

# # Bland-Altman Plots
# mean_linear, diff_linear, md_linear, sd_linear = bland_altman_stats(calculated_linear, odometry_linear)
# bland_altman_plot(mean_linear, diff_linear, md_linear, sd_linear, 'Bland-Altman Plot for Linear Velocities', 3)

# mean_angular, diff_angular, md_angular, sd_angular = bland_altman_stats(calculated_angular, odometry_angular)
# bland_altman_plot(mean_angular, diff_angular, md_angular, sd_angular, 'Bland-Altman Plot for Angular Velocities', 4)

# Error Analysis for Linear Velocities
errors_linear = calculated_linear - odometry_linear
error_analysis_and_plot(errors_linear, 'Linear Velocities', 5)

# Error Analysis for Angular Velocities
errors_angular = calculated_angular - odometry_angular
error_analysis_and_plot(errors_angular, 'Angular Velocities', 6)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Create figure for plotting
# plt.figure(figsize=(15, 10))

# # Plot World X Velocities over Time
# plt.subplot(3, 1, 1)
# # plt.plot(time_values, calculated_world_x, label='Calculated World Velocity X', marker='.', linestyle='-', markersize=5)
# plt.plot(time_values, odometry_world_x, label='Odometry World Velocity X', marker='.', linestyle='-', markersize=5)
# plt.title('World X Velocities Over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('World Velocity X (m/s)')
# plt.legend()
# plt.grid(True)

# # Plot World Y Velocities over Time
# plt.subplot(3, 1, 2)
# # plt.plot(time_values, calculated_world_y, label='Calculated World Velocity Y', marker='.', linestyle='-', markersize=5)
# plt.plot(time_values, odometry_world_y, label='Odometry World Velocity Y', marker='.', linestyle='-', markersize=5)
# plt.title('World Y Velocities Over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('World Velocity Y (m/s)')
# plt.legend()
# plt.grid(True)

# # Plot Angular Velocities over Time
# plt.subplot(3, 1, 3)
# # plt.plot(time_values, calculated_angular, label='Calculated Angular Velocity', marker='.', linestyle='-', markersize=5)
# plt.plot(time_values, odometry_angular, label='Odometry Angular Velocity', marker='.', linestyle='-', markersize=5)
# plt.title('Angular Velocities Over Time')
# plt.xlabel('Time (s)')
# plt.ylabel('Angular Velocity (rad/s)')
# plt.legend()
# plt.grid(True)

# # Adjust layout and show the plot
# plt.tight_layout()
# plt.show()