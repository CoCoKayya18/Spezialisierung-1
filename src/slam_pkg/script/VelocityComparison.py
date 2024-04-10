import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the complete filtered_dfset
filtered_dfframe = pd.read_csv('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/dataTesting/Filtered_Velocity_Data.csv')

# List all columns that you want to keep. Adjust this list based on your specific filtered_dfset and the columns you are interested in.
columns_of_interest = [
    'Time',
    'linear_velocity_x', 'angular_velocity_yaw',  # These might be computed or derived values
    'twist.twist.linear.x', 'twist.twist.angular.z',  # Example odometry velocities
    'linear.x', 'angular.z'  # Example commanded velocities
]

# Filter out the filtered_dfframe to only include the columns of interest
filtered_df = filtered_dfframe[columns_of_interest]

# # Drop any rows that may have NaN values in these columns if necessary
# filtered_df = filtered_df.dropna()

# # Save the filtered filtered_df to a new CSV file
# # filtered_df.to_csv('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/filtered_dfTesting/Filtered_Velocity_filtered_df.csv', index=False)

# print("Filtered velocity filtered_df saved successfully.")

# # Plotting
# plt.figure(figsize=(12, 8))

# # Linear velocities
# plt.subplot(3, 2, 1)  # 2 rows, 2 column, 1st subplot
# plt.plot(filtered_df['Time'].values, filtered_df['linear_velocity_x'].values, label='Calculated Linear Velocity', marker='o')
# plt.plot(filtered_df['Time'].values, filtered_df['twist.twist.linear.x'].values, label='Odometry Linear Velocity', marker='x')
# # plt.plot(filtered_df['Time'].values, filtered_df['linear.x'].values, label='Commanded Linear Velocity', marker='^')
# plt.title('Comparison of Linear Velocities')
# plt.xlabel('Time (s)')
# plt.ylabel('Linear Velocity (m/s)')
# plt.legend()
# plt.grid(True)

# # Angular velocities
# plt.subplot(3, 2, 2)  # 2 rows, 2 column, 2nd subplot
# plt.plot(filtered_df['Time'].values, filtered_df['angular_velocity_yaw'].values, label='Calculated Angular Velocity', marker='o')
# plt.plot(filtered_df['Time'].values, filtered_df['twist.twist.angular.z'].values, label='Odometry Angular Velocity', marker='x')
# # plt.plot(filtered_df['Time'].values, filtered_df['angular.z'].values, label='Commanded Angular Velocity', marker='^')
# plt.title('Comparison of Angular Velocities')
# plt.xlabel('Time (s)')
# plt.ylabel('Angular Velocity (rad/s)')
# plt.legend()
# plt.grid(True)

# plt.subplot(3, 2, 3)  # 2 rows, 2 column, 3nd subplot
# data1_mean = np.mean(filtered_df['linear_velocity_x'].values)
# data2_mean = np.mean(filtered_df['twist.twist.linear.x'].values)
# mean = np.mean([filtered_df['linear_velocity_x'], filtered_df['twist.twist.linear.x'].values], axis=0)
# diff = filtered_df['linear_velocity_x'] - filtered_df['twist.twist.linear.x'].values                   # Difference between data1 and data2
# md = np.mean(diff)                     # Mean of the difference
# sd = np.std(diff, axis=0)              # Standard deviation of the difference

# plt.scatter(mean, diff)
# plt.axhline(md, color='gray', linestyle='--')
# plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
# plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
# plt.title('Bland-Altman Plot X_Vel')
# plt.xlabel('Mean Value')
# plt.ylabel('Difference')
# plt.legend()
# plt.grid(True)

# plt.subplot(3, 2, 4)  # 2 rows, 2 column, 4nd subplot
# data1_mean = np.mean(filtered_df['angular_velocity_yaw'].values)
# data2_mean = np.mean(filtered_df['twist.twist.angular.z'].values)
# mean = np.mean([filtered_df['linear_velocity_x'], filtered_df['twist.twist.angular.z'].values], axis=0)
# diff = filtered_df['angular_velocity_yaw'].values - filtered_df['twist.twist.angular.z'].values                  # Difference between data1 and data2
# md = np.mean(diff)                     # Mean of the difference
# sd = np.std(diff, axis=0)              # Standard deviation of the difference

# plt.scatter(mean, diff)
# plt.axhline(md, color='gray', linestyle='--')
# plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
# plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
# plt.title('Bland-Altman Plot Yaw_Vel')
# plt.xlabel('Mean Value')
# plt.ylabel('Difference')
# plt.legend()
# plt.grid(True)

# plt.subplot(3, 2, 5)  # 2 rows, 2 column, 5th subplot
# errors = filtered_df['twist.twist.linear.x'].values - filtered_df['linear_velocity_x'].values
# mse = np.mean(np.square(errors))
# rmse = np.sqrt(mse)
# mae = np.mean(np.abs(errors))

# # Plot the errors
# plt.hist(errors, bins=50)
# plt.title('Error Distribution')
# plt.xlabel('Errors')
# plt.ylabel('Frequency')

# plt.scatter(filtered_df['linear_velocity_x'].values, errors)
# plt.title('Predicted vs Errors X_Vel')
# plt.xlabel('Predicted Values')
# plt.ylabel('Errors')
# plt.axhline(0, color='gray', linestyle='--')
# plt.show()

# plt.subplot(3, 2, 6)  # 2 rows, 2 column, 6th subplot
# errors = filtered_df['twist.twist.angular.z'].values - filtered_df['angular_velocity_yaw'].values
# mse = np.mean(np.square(errors))
# rmse = np.sqrt(mse)
# mae = np.mean(np.abs(errors))

# plt.hist(errors, bins=50)
# plt.title('Error Distribution')
# plt.xlabel('Errors')
# plt.ylabel('Frequency')

# plt.scatter(filtered_df['angular_velocity_yaw'].values, errors)
# plt.title('Predicted vs Errors')
# plt.xlabel('Predicted Values')
# plt.ylabel('Errors')
# plt.axhline(0, color='gray', linestyle='--')
# plt.show()

# # Show plot
# plt.tight_layout()
# plt.show()

calculated_linear = filtered_df['linear_velocity_x'].values
odometry_linear = filtered_df['twist.twist.linear.x'].values
calculated_angular = filtered_df['angular_velocity_yaw'].values
odometry_angular = filtered_df['twist.twist.angular.z'].values

# Calculate Bland-Altman plot statistics
def bland_altman_stats(data1, data2):
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    return mean, diff, md, sd

# Create a Bland-Altman plot
def bland_altman_plot(mean, diff, md, sd, title, subplot_index):
    plt.subplot(3, 2, subplot_index)
    plt.scatter(mean, diff, alpha=0.5)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
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

# Create figure for plotting
plt.figure(figsize=(15, 15))

# Plot linear velocities
plt.subplot(3, 2, 1)
plt.plot(filtered_df['Time'].values, calculated_linear, label='Calculated Linear Velocity', marker='o')
plt.plot(filtered_df['Time'].values, odometry_linear, label='Odometry Linear Velocity', marker='x')
plt.title('Comparison of Linear Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Linear Velocity (m/s)')
plt.legend()
plt.grid(True)

# Plot angular velocities
plt.subplot(3, 2, 2)
plt.plot(filtered_df['Time'].values, calculated_angular, label='Calculated Angular Velocity', marker='o')
plt.plot(filtered_df['Time'].values, odometry_angular, label='Odometry Angular Velocity', marker='x')
plt.title('Comparison of Angular Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

# Bland-Altman Plots
mean_linear, diff_linear, md_linear, sd_linear = bland_altman_stats(calculated_linear, odometry_linear)
bland_altman_plot(mean_linear, diff_linear, md_linear, sd_linear, 'Bland-Altman Plot for Linear Velocities', 3)

mean_angular, diff_angular, md_angular, sd_angular = bland_altman_stats(calculated_angular, odometry_angular)
bland_altman_plot(mean_angular, diff_angular, md_angular, sd_angular, 'Bland-Altman Plot for Angular Velocities', 4)

# Error Analysis for Linear Velocities
errors_linear = calculated_linear - odometry_linear
error_analysis_and_plot(errors_linear, 'Linear Velocities', 5)

# Error Analysis for Angular Velocities
errors_angular = calculated_angular - odometry_angular
error_analysis_and_plot(errors_angular, 'Angular Velocities', 6)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()