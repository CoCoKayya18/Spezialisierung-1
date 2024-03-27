import pandas as pd
import numpy as np



# Function to calculate the next pose of the robot
def calculate_new_pose(vc, wc, theta, dt):
    if wc == 0:
        # The robot is moving in a straight line
        delta_x = vc * dt
        delta_y = 0.0
        delta_theta = 0.0
    else:
        # The robot is rotating around a circle with radius R = vc / wc
        delta_x = (vc / wc) * (np.sin(theta + wc * dt) - np.sin(theta))
        delta_y = (vc / wc) * (-np.cos(theta + wc * dt) + np.cos(theta))
        delta_theta = wc * dt
    
    return delta_x, delta_y, delta_theta


datafilepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'

# Load the data
velocities_df = pd.read_csv(datafilepath)

# Sort by time to ensure correct sequential processing
velocities_df.sort_values(by='Time', inplace=True)

# Initialize columns for deltas
velocities_df['calculated_delta_x'] = 0.0
velocities_df['calculated_delta_y'] = 0.0
velocities_df['calculated_delta_yaw'] = 0.0

# Assume starting at (0, 0) with a yaw of 0
x, y, theta = 0.0, 0.0, 0.0

# Calculate the deltas
for index, row in velocities_df.iterrows():

    vc = row['linear_velocity_x']
    wc = row['angular_velocity_yaw']

    dt = velocities_df.loc[index, 'Time'] - velocities_df.loc[index - 1, 'Time']
    
    # Calculate deltas
    delta_x, delta_y, delta_theta = calculate_new_pose(vc, wc, theta, dt)

    calculatedDeltas_df = pd.DataFrame(delta_x, delta_y, delta_theta)

# Save to a new CSV file
calculatedDeltas_df.to_csv('deltas.csv', index=False)
print("Deltas have been calculated and saved to 'deltas.csv'.")