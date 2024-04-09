import pandas as pd
import matplotlib.pyplot as plt

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

# Drop any rows that may have NaN values in these columns if necessary
filtered_df = filtered_df.dropna()

# Save the filtered filtered_df to a new CSV file
# filtered_df.to_csv('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/filtered_dfTesting/Filtered_Velocity_filtered_df.csv', index=False)

print("Filtered velocity filtered_df saved successfully.")

# Plotting
plt.figure(figsize=(12, 8))

# Linear velocities
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.plot(filtered_df['Time'].values, filtered_df['linear_velocity_x'].values, label='Calculated Linear Velocity', marker='o')
plt.plot(filtered_df['Time'].values, filtered_df['twist.twist.linear.x'].values, label='Odometry Linear Velocity', marker='x')
plt.plot(filtered_df['Time'].values, filtered_df['linear.x'].values, label='Commanded Linear Velocity', marker='^')
plt.title('Comparison of Linear Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Linear Velocity (m/s)')
plt.legend()
plt.grid(True)

# Angular velocities
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(filtered_df['Time'].values, filtered_df['angular_velocity_yaw'].values, label='Calculated Angular Velocity', marker='o')
plt.plot(filtered_df['Time'].values, filtered_df['twist.twist.angular.z'].values, label='Odometry Angular Velocity', marker='x')
plt.plot(filtered_df['Time'].values, filtered_df['angular.z'].values, label='Commanded Angular Velocity', marker='^')
plt.title('Comparison of Angular Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
