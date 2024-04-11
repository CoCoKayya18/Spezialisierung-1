import rosbag
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_positions_from_bag(bag_file, topic_name):
    """Reads ground truth positions from a ROS bag file."""
    positions = []
    with rosbag.Bag(bag_file, 'r') as bag:
        for _, msg, _ in bag.read_messages(topics=[topic_name]):
            positions.append((msg.pose.pose.position.x, msg.pose.pose.position.y))
    return positions

def calculate_trajectory_from_deltas(deltas_csv_path, start_x=0.5, start_y=0.5):
    deltas_df = pd.read_csv(deltas_csv_path)
    deltas_df.sort_values(by='Time', inplace=True)  # Ensure data is sorted by time, important if timestamps can be disordered.

    # Initialize the first position and tracking variables
    trajectory_x = [start_x]
    trajectory_y = [start_y]
    current_x = start_x
    current_y = start_y
    last_time = -np.inf  # Initialize with a value that's always lower than the first timestamp

    # Process deltas
    for i, row in deltas_df.iterrows():
        if row['Time'] < last_time:  # If the current time is less than the last time, reset starting position
            trajectory_x.append(current_x)
            trajectory_y.append(current_y)
            current_x = start_x
            current_y = start_y

        current_x += row['delta_position_x']
        current_y += row['delta_position_y']
        trajectory_x.append(current_x)
        trajectory_y.append(current_y)
        last_time = row['Time']

    return list(zip(trajectory_x, trajectory_y))

# Adjusted plot_trajectories function
def plot_trajectories(ground_truth, calculated):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})

    # Unzip tuples for plotting
    ground_truth_x, ground_truth_y = zip(*ground_truth)
    calculated_x, calculated_y = zip(*calculated)  # Unzip calculated if it's a list of tuples

    axes[0, 0].plot(ground_truth_x, ground_truth_y, 'b-', marker='.', markersize=1, label='Ground Truth')
    axes[0, 0].scatter(start_x, start_y, color='green', s=10, label='Start')
    axes[0, 0].set_title('Ground Truth Trajectory')
    axes[0, 0].legend()

    axes[0, 1].plot(calculated_x, calculated_y, 'r--', marker='.', markersize=1, label='Calculated')
    axes[0, 1].scatter(start_x, start_y, color='green', s=10)
    axes[0, 1].set_title('Calculated Trajectory')
    axes[0, 1].legend()

    # Calculate positional errors
    errors = [np.linalg.norm(np.array(calculated[i]) - np.array(ground_truth[i])) for i in range(min(len(calculated), len(ground_truth)))]

    axes[1, 0].hist(errors, bins=50, alpha=0.5)
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].set_xlabel('Error (m)')
    axes[1, 0].set_ylabel('Frequency')

    axes[1, 1].axis('off')  # Hide the unused subplot

    plt.tight_layout()
    plt.show()

# Paths to your ROS bag file and deltas CSV file
deltas_csv_path = ['/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data_ALLSet1.csv', '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data_ALLSet2.csv', '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data_ALLSet3.csv', '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data_ALLSet4.csv']
ground_truth_topic_name = 'ground_truth/state'

bag_files = ['/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/AllSensor_data_2024-04-10-15-18-21.bag', '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/AllSensor_data_2024-04-10-15-15-44.bag', '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/AllSensor_data_2024-04-10-15-03-50.bag', '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/AllSensor_data_2024-04-10-15-03-04.bag']
    
all_ground_truth_positions = []
all_calculated_positions = []

# Loop through each bag file and append positions
for bag_file_path in bag_files:
    ground_truth_positions = read_positions_from_bag(bag_file_path, ground_truth_topic_name)
    all_ground_truth_positions.extend(ground_truth_positions)

# Initial position (change if needed)
start_x, start_y = 0.5, 0.5

# Read the ground truth positions from the ROS bag

# Calculate the trajectory from the deltas CSV file

for delta_paths in deltas_csv_path:
    calculated_trajectory = calculate_trajectory_from_deltas(delta_paths, start_x, start_y)
    all_calculated_positions.extend(calculated_trajectory)

ground_truth_x, ground_truth_y = zip(*all_ground_truth_positions)

# Plot the trajectories
plot_trajectories(all_ground_truth_positions, all_calculated_positions)