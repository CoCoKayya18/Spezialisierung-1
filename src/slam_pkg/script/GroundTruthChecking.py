import rosbag
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to read ground truth positions from a ROS bag
def read_positions_from_bag(bag_file, topic_name):
    positions = []
    
    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, _ in bag.read_messages(topics=[topic_name]):
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            positions.append((x, y))
            
    return positions

# Function to calculate the trajectory from deltas in a CSV file
def calculate_trajectory_from_deltas(deltas_csv_path, start_x=0.5, start_y=0.5):
    deltas_df = pd.read_csv(deltas_csv_path)
    # Initialize the first position
    trajectory_x = [start_x]
    trajectory_y = [start_y]
    
    for i in range(len(deltas_df)):
        trajectory_x.append(trajectory_x[-1] + deltas_df.iloc[i]['delta_position_x'])
        trajectory_y.append(trajectory_y[-1] + deltas_df.iloc[i]['delta_position_y'])
        
    return trajectory_x, trajectory_y

# Function to plot the robot's motion on an XY plane
def plot_trajectories(ground_truth, calculated):
    plt.figure(figsize=(12, 6))

    # Plot the ground truth trajectory
    ground_truth_x, ground_truth_y = zip(*ground_truth)
    plt.plot(ground_truth_x, ground_truth_y, label='Ground Truth Trajectory', color='blue', marker='.')
    
    # Plot the calculated trajectory from positional changes
    plt.plot(calculated[0], calculated[1], label='Calculated Trajectory', color='red', linestyle='--', marker='x')

    # Mark the starting position
    plt.scatter(start_x, start_y, color='green', label='Start', zorder=5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Robot Trajectories on XY Plane')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Paths to your ROS bag file and deltas CSV file
deltas_csv_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data_ALLSet.csv'
ground_truth_topic_name = 'ground_truth/state'

bag_files = ['/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/AllSensor_data_2024-04-10-15-18-21.bag', '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/AllSensor_data_2024-04-10-15-03-50.bag', '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/AllSensor_data_2024-04-10-15-15-44.bag', '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/AllSensor_data_2024-04-10-15-03-04.bag']
ground_truth_positions = []
for bag_file in bag_files:
    ground_truth_positions = ground_truth_positions + read_positions_from_bag(bag_file, ground_truth_topic_name)

# Initial position (change if needed)
start_x, start_y = 0.5, 0.5

# Read the ground truth positions from the ROS bag

# Calculate the trajectory from the deltas CSV file
calculated_trajectory = calculate_trajectory_from_deltas(deltas_csv_path, start_x, start_y)

# Plot the trajectories
plot_trajectories(ground_truth_positions, calculated_trajectory)
