import pandas as pd
from bagpy import bagreader
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

class BagDataProcessor:
    def __init__(self, bagfilepath):
        self.bag = bagreader(bagfilepath)
        self.filepath = bagfilepath

    def read_topic_to_dataframe(self, topic_name):
        csv_path = self.bag.message_by_topic(topic_name)
        return pd.read_csv(csv_path) if csv_path else pd.DataFrame()

    def calculate_ground_truth_deltas(self, df):
        # Convert seconds and nanoseconds to a single time column in seconds
        df['Time'] = pd.to_numeric(df['header.stamp.secs']) + pd.to_numeric(df['header.stamp.nsecs']) * 1e-9
        df['delta_position_x'] = df['pose.pose.position.x'].diff().fillna(0) # Get Delta_X
        df['delta_position_y'] = df['pose.pose.position.y'].diff().fillna(0) # Get Delta_Y
        # Convert quaternion to Euler angles (yaw)
        quaternions = df[['pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w']].to_numpy()
        eulers = R.from_quat(quaternions).as_euler('xyz', degrees=False)  # xyz order, output in radians
        df['yaw'] = eulers[:, 2]  # z-axis (yaw)
        
        # Calculate yaw delta and normalize
        df['delta_yaw'] = df['yaw'].diff().fillna(0) # Get Delta_Yaw
        df['delta_yaw'] = np.arctan2(np.sin(df['delta_yaw']), np.cos(df['delta_yaw']))  # Normalize the yaw delta

        return df[['Time', 'delta_position_x', 'delta_position_y', 'delta_yaw']]

    def calculate_joint_velocities_and_accelerations(self, df):
        # Convert seconds and nanoseconds to a single time column in seconds.
        df['Time'] = pd.to_numeric(df['header.stamp.secs']) + pd.to_numeric(df['header.stamp.nsecs']) * 1e-9
        
        wheel_radius = 0.066/2  # meter
        wheel_base = 0.160  # distance between wheels in meters

        # Calculate time differences and replace any zeros with NaN to avoid division by zero
        # time_diffs = df['Time'].diff().replace(0, np.nan)
        time_diffs = 0.0333

        # Calculate linear and angular velocities
        df['linear_velocity_x'] = ((df['position_0'].diff() + df['position_1'].diff()) / 2) / time_diffs * wheel_radius
        df['angular_velocity_yaw'] = ((df['position_0'].diff() - df['position_1'].diff()) / wheel_base) / time_diffs * wheel_radius
        
        # Calculate accelerations
        df['linear_acceleration_x'] = df['linear_velocity_x'].diff() / time_diffs
        df['angular_acceleration_yaw'] = df['angular_velocity_yaw'].diff() / time_diffs
        
        return df[['Time', 'linear_velocity_x', 'angular_velocity_yaw', 'linear_acceleration_x', 'angular_acceleration_yaw']]

    def process_and_save_data(self, ground_truth_df, joint_state_df):
        processed_gt_df = self.calculate_ground_truth_deltas(ground_truth_df)
        processed_joint_df = self.calculate_joint_velocities_and_accelerations(joint_state_df)

        dataFilePathDeltas = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/GT_Deltas.csv'
        dataFilePathVelsAndAccs = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Vels_And_Accels.csv'
        mergedPath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'

        # Remove rows with any NaN values (which now includes the original 'inf' values)
        processed_gt_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        processed_gt_df = processed_gt_df.dropna()
        processed_joint_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        processed_joint_df = processed_joint_df.dropna()

        processed_gt_df.to_csv(dataFilePathDeltas, index=False)
        processed_joint_df.to_csv(dataFilePathVelsAndAccs, index=False)

        processed_gt_df = processed_gt_df.sort_values(by='Time')
        processed_joint_df = processed_joint_df.sort_values(by='Time')

        # Use merge_asof to align the data based on the 'Time' column
        combined_df = pd.merge_asof(processed_gt_df, processed_joint_df, on='Time')

        # Remove first two lines
        combined_df = combined_df.iloc[2:].reset_index(drop=True)

        # Save the combined DataFrame to a CSV file
        combined_df.to_csv(mergedPath, index=False)

        return processed_gt_df, processed_joint_df

def process_bag_file(bag_file_path):
    processor = BagDataProcessor(bag_file_path)
    ground_truth_df = processor.read_topic_to_dataframe('ground_truth/state')
    joint_state_df = processor.read_topic_to_dataframe('joint_states')

    if ground_truth_df.empty or joint_state_df.empty:
        print("One or more of the topics do not exist in the bag file or are empty.")
        return

    processed_gt_df, processed_joint_df = processor.process_and_save_data(ground_truth_df, joint_state_df)

if __name__ == '__main__':
    bag_files = ['/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/rosbag_data_2024-03-26-21-09-55.bag', '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/rosbag_data_2024-03-27-12-30-11.bag']
    for bag_file in bag_files:
        process_bag_file(bag_file)
