import pandas as pd
from bagpy import bagreader
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import math
import glob

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
        df['delta_position_x_world'] = df['pose.pose.position.x'].diff().fillna(0)  # Get Delta_X
        df['delta_position_y_world'] = df['pose.pose.position.y'].diff().fillna(0)  # Get Delta_Y
        
        # Convert quaternion to Euler angles (yaw)
        quaternions = df[['pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w']].to_numpy()
        eulers = R.from_quat(quaternions).as_euler('xyz', degrees=False)  # xyz order, output in radians
        df['yaw_world'] = eulers[:, 2]  # z-axis (yaw)
        
        # Calculate yaw delta and normalize
        df['delta_yaw'] = df['yaw_world'].diff().fillna(0)  # Get Delta_Yaw
        df['delta_yaw'] = np.arctan2(np.sin(df['delta_yaw']), np.cos(df['delta_yaw']))  # Normalize the yaw delta

        return df[['Time', 'yaw_world', 'delta_position_x_world', 'delta_position_y_world', 'delta_yaw']]

    def calculate_joint_velocities_and_accelerations(self, df, ground_truth_df, initialOrientation):
        # Convert seconds and nanoseconds to a single time column in seconds
        df['Time'] = pd.to_numeric(df['header.stamp.secs']) + pd.to_numeric(df['header.stamp.nsecs']) * 1e-9
        ground_truth_df['Time'] = pd.to_numeric(ground_truth_df['header.stamp.secs']) + pd.to_numeric(ground_truth_df['header.stamp.nsecs']) * 1e-9

        # Merge ground truth and joint states based on time
        merged_df = pd.merge_asof(df, ground_truth_df[['Time', 'yaw_world']], on='Time')
        
        wheel_radius = 0.066 / 2  # meter
        wheel_base = 0.160  # distance between wheels in meters
        time_diffs = 0.0333

        # Calculate linear and angular velocities
        df['linear_velocity_x'] = (((df['position_0'].diff() + df['position_1'].diff()) / 2) / time_diffs) * wheel_radius
        df['angular_velocity_yaw'] = (((df['position_0'].diff() - df['position_1'].diff()) / wheel_base) / time_diffs) * wheel_radius

        df['angular_velocity_yaw'].fillna(method='bfill', inplace=True)  # Backward fill for the first NaN
        df['angular_velocity_yaw'].interpolate(inplace=True)  # Interpolate remaining NaNs if any

        theta = initialOrientation + np.cumsum(np.insert(df['angular_velocity_yaw'].values, 0, 0)[:-1]) * time_diffs
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        corrected_thetas = []
        for idx, row in merged_df.iterrows():
            calculated_yaw = theta[idx]
            ground_truth_yaw = row['yaw_world']

            # Check for large discrepancies and correct if necessary
            if abs(calculated_yaw - ground_truth_yaw) > np.pi / 4:
                corrected_yaw = ground_truth_yaw
            else:
                corrected_yaw = calculated_yaw

            corrected_thetas.append(corrected_yaw)

        df['Theta_calculated'] = corrected_thetas

        df['world_velocity_x'] = df['linear_velocity_x'] * np.cos(df['Theta_calculated'])
        df['world_velocity_y'] = df['linear_velocity_x'] * np.sin(df['Theta_calculated'])
        
        # Calculate accelerations
        df['linear_acceleration_x'] = df['linear_velocity_x'].diff() / time_diffs
        df['angular_acceleration_yaw'] = df['angular_velocity_yaw'].diff() / time_diffs
        
        return df[['Time', 'Theta_calculated', 'linear_velocity_x', 'world_velocity_x', 'world_velocity_y', 'angular_velocity_yaw', 'linear_acceleration_x', 'angular_acceleration_yaw']]
    
    def calculate_kinematic_deltas(self, df, initialTheta):
        df['kinematic_delta_x'] = 0.0
        df['kinematic_delta_y'] = 0.0
        df['kinematic_delta_yaw'] = 0.0
        
        # Initial pose
        x, y, theta = 0.0, 0.0, initialTheta
        
        for index, row in df.iterrows():
            if index == 0:
                continue
            vc = row['linear_velocity_x']
            wc = row['angular_velocity_yaw']
            dt = 0.0333
            
            # Skip the first row or any rows where dt <= 0
            if dt <= 0:
                continue
                
            delta_x, delta_y, delta_theta = self.calculate_robot_pose_change(vc, wc, theta, dt)
            
            df.at[index, 'kinematic_delta_x'] = delta_x
            df.at[index, 'kinematic_delta_y'] = delta_y
            df.at[index, 'kinematic_delta_yaw'] = delta_theta
            
            # Update the pose
            x += delta_x
            y += delta_y
            theta += delta_theta
            theta = np.arctan2(np.sin(theta), np.cos(theta))  # Normalize Theta
        
        return df

    def calculate_robot_pose_change(self, vc, wc, theta, dt):
        # Kinematic model for differential drive robot
        if wc == 0:
            # Straight line movement
            calculated_delta_x = vc * dt
            calculated_delta_y = 0.0
        else:
            # Arc movement
            calculated_delta_x = (vc / wc) * (np.sin(theta + wc * dt) - np.sin(theta))
            calculated_delta_y = (vc / wc) * (-np.cos(theta + wc * dt) + np.cos(theta))
        calculated_delta_theta = wc * dt

        return calculated_delta_x, calculated_delta_y, calculated_delta_theta

    def process_and_save_data(self, ground_truth_df, joint_state_df, odom_df, cmdVel_df, imu_df, counter, initialOrientation, folderName, single=False):
        processed_gt_df = self.calculate_ground_truth_deltas(ground_truth_df)
        processed_joint_df = self.calculate_joint_velocities_and_accelerations(joint_state_df, ground_truth_df, initialOrientation)

        suffix = "_single" if single else ""
        data_dir = f'../Spezialisierung-1/src/slam_pkg/data/{folderName}{suffix}'
        dataFilePathDeltas = os.path.join(data_dir, f'GT_Deltas{suffix}.csv')
        dataFilePathVelsAndAccs = os.path.join(data_dir, f'Vels_And_Accels{suffix}.csv')
        mergedPath = os.path.join(data_dir, f'FullData{suffix}.csv')

        os.makedirs(data_dir, exist_ok=True)

        # Remove rows with any NaN values (which now includes the original 'inf' values)
        processed_gt_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        processed_gt_df = processed_gt_df.dropna()
        processed_joint_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        processed_joint_df = processed_joint_df.dropna()

        # Calculate the kinematic Delta_X too and add to DF
        processed_joint_df = self.calculate_kinematic_deltas(processed_joint_df, initialOrientation)

        processed_gt_df.to_csv(dataFilePathDeltas, index=False)
        processed_joint_df.to_csv(dataFilePathVelsAndAccs, index=False)

        processed_gt_df = processed_gt_df.sort_values(by='Time')
        processed_joint_df = processed_joint_df.sort_values(by='Time')

        # Use merge_asof to align the data based on the 'Time' column
        combined_df = pd.merge_asof(processed_gt_df, processed_joint_df, on='Time')
        combined_df = pd.merge_asof(combined_df, odom_df, on='Time')
        combined_df = pd.merge_asof(combined_df, cmdVel_df, on='Time')
        combined_df = pd.merge_asof(combined_df, imu_df, on='Time')

        columns_of_interest = ['Theta_calculated', 'yaw_world', 'linear_velocity_x', 'world_velocity_x', 'world_velocity_y', 'angular_velocity_yaw', 'linear_acceleration_x', 'angular_acceleration_yaw', 'delta_position_x_world', 'delta_position_y_world', 'delta_yaw', 'kinematic_delta_x', 'kinematic_delta_y', 'kinematic_delta_yaw']

        missing_values = combined_df[columns_of_interest].isnull().sum()
        if missing_values.any():
            print("Missing values detected in columns of interest:")
            print(missing_values)
            # Drop rows where columns of interest have missing values
            combined_df = combined_df.dropna(subset=columns_of_interest)
        else:
            print("No missing values detected in columns of interest.")

        # Remove first two lines
        combined_df = combined_df.iloc[2:].reset_index(drop=True)

        if os.path.exists(mergedPath) and not single:
            # If it does, read the existing data and append the new data
            existing_data = pd.read_csv(mergedPath)
            combined_df = pd.concat([existing_data, combined_df], ignore_index=True)

        # Save the combined DataFrame to a CSV file
        combined_df.to_csv(mergedPath, index=False)

        return processed_gt_df, processed_joint_df
    

def process_bag_file(bag_file_path, counter, initialOrientation, folderName, single=False):
    Incounter = counter
    processor = BagDataProcessor(bag_file_path)
    ground_truth_df = processor.read_topic_to_dataframe('ground_truth/state')
    joint_state_df = processor.read_topic_to_dataframe('joint_states')
    odom_df = processor.read_topic_to_dataframe('odom')
    cmdVel_df = processor.read_topic_to_dataframe('cmd_vel')
    imu_df = processor.read_topic_to_dataframe('imu')

    if ground_truth_df.empty or joint_state_df.empty or odom_df.empty or cmdVel_df.empty or imu_df.empty:
        print("One or more of the topics do not exist in the bag file or are empty.")
        return

    processed_gt_df, processed_joint_df = processor.process_and_save_data(ground_truth_df, joint_state_df, odom_df, cmdVel_df, imu_df, Incounter, initialOrientation, folderName, single)


def get_bag_files(directory):
    directory = os.path.abspath(directory)
    bag_files = glob.glob(os.path.join(directory, '*.bag'))
    print(bag_files)
    return bag_files

if __name__ == '__main__':
    directories = [
        '../Spezialisierung-1/src/slam_pkg/rosbag_files/x_direction_positive',
        '../Spezialisierung-1/src/slam_pkg/rosbag_files/x_direction_negative',
        '../Spezialisierung-1/src/slam_pkg/rosbag_files/y_direction_positive',
        '../Spezialisierung-1/src/slam_pkg/rosbag_files/y_direction_negative',
        '../Spezialisierung-1/src/slam_pkg/rosbag_files/diagonal_first_quad',
        '../Spezialisierung-1/src/slam_pkg/rosbag_files/diagonal_second_quad',
        '../Spezialisierung-1/src/slam_pkg/rosbag_files/diagonal_third_quad',
        '../Spezialisierung-1/src/slam_pkg/rosbag_files/diagonal_fourth_quad'
        '../Spezialisierung-1/src/slam_pkg/rosbag_files/random',
        '../Spezialisierung-1/src/slam_pkg/rosbag_files/random2',
        '../Spezialisierung-1/src/slam_pkg/rosbag_files/random3'
    ]

    for directory_path in directories:
        bag_files = get_bag_files(directory_path)
        counter = 1
        folderName = os.path.basename(directory_path)

        if folderName == 'x_direction_positive':
            initialTheta = 0.0
        elif folderName == 'x_direction_negative':
            initialTheta = np.pi
        elif folderName == 'y_direction_positive':
            initialTheta = np.pi / 2
        elif folderName == 'y_direction_negative':
            initialTheta = -np.pi / 2
        elif folderName == 'diagonal_first_quad':
            initialTheta = np.pi / 4
        elif folderName == 'diagonal_second_quad':
            initialTheta = 3 * np.pi / 4
        elif folderName == 'diagonal_third_quad':
            initialTheta = -3 * np.pi / 4
        elif folderName == 'diagonal_fourth_quad':
            initialTheta = -np.pi / 4
        elif folderName == 'random':
            initialTheta = 0
        elif folderName == 'random2':
            initialTheta = 0
        elif folderName == 'random3':
            initialTheta = 0

        for bag_file in bag_files:
            print(f"Processing {bag_file} with initialTheta {initialTheta}")
            process_bag_file(bag_file, counter, initialTheta, folderName)
            counter += 1

        # Process a single file for each direction
        if bag_files:
            single_bag_file = bag_files[0]
            print(f"Processing single file {single_bag_file} with initialTheta {initialTheta}")
            process_bag_file(single_bag_file, 1, initialTheta, folderName, single=True)
