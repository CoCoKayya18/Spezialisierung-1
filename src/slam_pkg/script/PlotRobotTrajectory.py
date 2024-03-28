import pandas as pd
import matplotlib.pyplot as plt

# Load your DataFrame
DataFilePath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data_Testing.csv'

df = pd.read_csv(DataFilePath)

# Compute the cumulative sum of the deltas for kinematic and ground truth
df['cumulative_kinematic_x'] = df['kinematic_delta_x'].cumsum()
df['cumulative_kinematic_y'] = df['kinematic_delta_y'].cumsum()
df['cumulative_ground_truth_x'] = df['delta_position_x'].cumsum()
df['cumulative_ground_truth_y'] = df['delta_position_y'].cumsum()

# Convert Series to numpy arrays
cumulative_kinematic_x = df['cumulative_kinematic_x'].to_numpy()
cumulative_kinematic_y = df['cumulative_kinematic_y'].to_numpy()
cumulative_ground_truth_x = df['cumulative_ground_truth_x'].to_numpy()
cumulative_ground_truth_y = df['cumulative_ground_truth_y'].to_numpy()

# Plotting
plt.figure(figsize=(12, 6))

# Plot the kinematic path
plt.plot(cumulative_kinematic_x, cumulative_kinematic_y, label='Kinematic Path', marker='o', linestyle='-', markersize=2)

# Plot the ground truth path
plt.plot(cumulative_ground_truth_x, cumulative_ground_truth_y, label='Ground Truth Path', marker='x', linestyle='--', markersize=2)

# Adding titles and labels
plt.title('Robot Trajectory: Kinematic vs. Ground Truth')
plt.xlabel('Cumulative Delta X')
plt.ylabel('Cumulative Delta Y')

# Adding a legend
plt.legend()

# Show grid
plt.grid(True)

# Display the plot
plt.show()
