import pandas as pd

# Assuming df is your DataFrame

# Define maximum realistic values
max_linear_velocity = 0.22  # m/s
max_angular_velocity = 2.84  # rad/s
max_linear_acceleration = 2.5  # m/s^2, assumed value for filtering
max_angular_acceleration = 3.2  # rad/s^2, assumed value for filtering

dataFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'
FliteredDataFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'

df = pd.read_csv(dataFilePath)

# Filter out unrealistic data points
realistic_df = df[(df['linear_velocity_x'].abs() <= max_linear_velocity) &
                  (df['angular_velocity_yaw'].abs() <= max_angular_velocity) & 
                  (df['linear_acceleration_x'].abs() <= max_linear_acceleration) &
                  (df['angular_acceleration_yaw'].abs() <= max_angular_acceleration)]

realistic_df.to_csv(FliteredDataFilePath)