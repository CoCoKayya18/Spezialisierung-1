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
        df['delta_position_x_world'] = df['pose.pose.position.x'].diff().fillna(0) # Get Delta_X
        df['delta_position_y_world'] = df['pose.pose.position.y'].diff().fillna(0) # Get Delta_Y
        
        # Convert quaternion to Euler angles (yaw)
        quaternions = df[['pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w']].to_numpy()
        eulers = R.from_quat(quaternions).as_euler('xyz', degrees=False)  # xyz order, output in radians
        df['yaw_world'] = eulers[:, 2]  # z-axis (yaw)
        
        # Calculate yaw delta and normalize
        df['delta_yaw'] = df['yaw_world'].diff().fillna(0) # Get Delta_Yaw
        df['delta_yaw'] = np.arctan2(np.sin(df['delta_yaw']), np.cos(df['delta_yaw']))  # Normalize the yaw delta

        # Transform World Frame Deltas to Robot Frame Deltas
        # cos_yaw = np.cos(df['yaw'])
        # sin_yaw = np.sin(df['yaw'])
        # df['delta_position_x_robot'] = cos_yaw * df['delta_position_x'] + sin_yaw * df['delta_position_y']
        # df['delta_position_y_robot'] = -sin_yaw * df['delta_position_x'] + cos_yaw * df['delta_position_y']        

        return df[['Time', 'yaw_world', 'delta_position_x_world', 'delta_position_y_world', 'delta_yaw']]
        # return df[['Time', 'yaw', 'delta_position_x_robot', 'delta_position_y_robot', 'delta_yaw']]

    def calculate_joint_velocities_and_accelerations(self, df, initialOrientation):
        # Convert seconds and nanoseconds to a single time column in seconds.
        df['Time'] = pd.to_numeric(df['header.stamp.secs']) + pd.to_numeric(df['header.stamp.nsecs']) * 1e-9
        
        wheel_radius = 0.066/2  # meter
        wheel_base = 0.160  # distance between wheels in meters
        time_diffs = 0.0333


        # print(df['position_0'].diff() + df['position_1'].diff())

        # Calculate linear and angular velocities
        df['linear_velocity_x'] = (((df['position_0'].diff() + df['position_1'].diff()) / 2) / time_diffs) * wheel_radius
        df['angular_velocity_yaw'] = (((df['position_0'].diff() - df['position_1'].diff()) / wheel_base) / time_diffs) * wheel_radius

        df['angular_velocity_yaw'].fillna(method='bfill', inplace=True)  # Backward fill for the first NaN
        df['angular_velocity_yaw'].interpolate(inplace=True)  # Interpolate remaining NaNs if any

        theta = initialOrientation + np.cumsum(np.insert(df['angular_velocity_yaw'].values, 0, 0)[:-1]) * time_diffs
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        df['Theta_calculated'] = theta

        # print(theta)

        df['world_velocity_x'] = df['linear_velocity_x'] * np.cos(theta)
        df['world_velocity_y'] = df['linear_velocity_x'] * np.sin(theta)
        
        # Calculate accelerations
        df['linear_acceleration_x'] = df['linear_velocity_x'].diff() / time_diffs
        df['angular_acceleration_yaw'] = df['angular_velocity_yaw'].diff() / time_diffs
        
        # return df[['Time', 'Theta', 'linear_velocity_x', 'angular_velocity_yaw', 'linear_acceleration_x', 'angular_acceleration_yaw']]
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
            # print(f'Theta unnormalized: {theta}')
            theta = np.arctan2(np.sin(theta), np.cos(theta)) # Normalize Theta
            # print(f'Theta normalized: {theta}')
        
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

    def process_and_save_data(self, ground_truth_df, joint_state_df, odom_df, cmdVel_df, imu_df, counter, initialOrientation, folderName):
        processed_gt_df = self.calculate_ground_truth_deltas(ground_truth_df)
        processed_joint_df = self.calculate_joint_velocities_and_accelerations(joint_state_df, initialOrientation)

        # SpecialCase = folderName
        # SpecialCase = ''
        
        data_dir = f'../Spezialisierung-1/src/slam_pkg/data/{folderName}'
        dataFilePathDeltas = os.path.join(data_dir, 'GT_Deltas.csv')
        dataFilePathVelsAndAccs = os.path.join(data_dir, 'Vels_And_Accels.csv')
        mergedPath = os.path.join(data_dir, 'FullData.csv')

        os.makedirs(data_dir, exist_ok=True)

        # dataFilePathDeltas = f'../Spezialisierung-1/src/slam_pkg/data/{SpecialCase}/GT_Deltas.csv'
        # dataFilePathVelsAndAccs = f'../Spezialisierung-1/src/slam_pkg/data/{SpecialCase}/Vels_And_Accels.csv'
        # mergedPath = f'../Spezialisierung-1/src/slam_pkg/data/{SpecialCase}/FullData.csv'

        # dataFilePathDeltas = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/GT_Deltas{SpecialCase}{counter}.csv'
        # dataFilePathVelsAndAccs = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Vels_And_Accels{SpecialCase}{counter}.csv'
        # mergedPath = f'/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data{SpecialCase}{counter}.csv'

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

        # columns_of_interest = ['Theta', 'yaw', 'linear_velocity_x', 'angular_velocity_yaw', 'linear_acceleration_x', 'angular_acceleration_yaw', 'delta_position_x_robot', 'delta_position_y_robot', 'delta_yaw', 'kinematic_delta_x', 'kinematic_delta_y', 'kinematic_delta_yaw']
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

        if os.path.exists(mergedPath):
            # If it does, read the existing data and append the new data
            existing_data = pd.read_csv(mergedPath)
            combined_df = pd.concat([existing_data, combined_df], ignore_index=True)

        # Save the combined DataFrame to a CSV file
        combined_df.to_csv(mergedPath, index=False)

        return processed_gt_df, processed_joint_df
    


def process_bag_file(bag_file_path, counter, initialOrientation, folderName):
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

    processed_gt_df, processed_joint_df = processor.process_and_save_data(ground_truth_df, joint_state_df, odom_df, cmdVel_df, imu_df, Incounter, initialOrientation, folderName)


def get_bag_files(directory):
    directory = os.path.abspath(directory)
    bag_files = glob.glob(os.path.join(directory, '*.bag'))
    return bag_files

def combine_csv_files(paths, filenames, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in filenames:
        combined_df = pd.DataFrame()
        
        for path in paths:
            file_path = os.path.join(path, filename)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            else:
                print(f"File {file_path} does not exist.")
        
        # Save the combined DataFrame to the output directory
        output_file_path = os.path.join(output_dir, filename)
        combined_df.to_csv(output_file_path, index=False)
        print(f"Combined file saved to {output_file_path}")

if __name__ == '__main__':
    
    # directory_path = '../Spezialisierung-1/src/slam_pkg/rosbag_files/x_direction_positive'
    # directory_path = '../Spezialisierung-1/src/slam_pkg/rosbag_files/x_direction_negative'
    # directory_path = '../Spezialisierung-1/src/slam_pkg/rosbag_files/y_direction_positive'
    # directory_path = '../Spezialisierung-1/src/slam_pkg/rosbag_files/y_direction_negative'
    # directory_path = '../Spezialisierung-1/src/slam_pkg/rosbag_files/diagonal_first_quad'
    # directory_path = '../Spezialisierung-1/src/slam_pkg/rosbag_files/diagonal_second_quad'
    # directory_path = '../Spezialisierung-1/src/slam_pkg/rosbag_files/diagonal_third_quad'
    # directory_path = '../Spezialisierung-1/src/slam_pkg/rosbag_files/diagonal_fourth_quad'
    # bag_files = get_bag_files(directory_path)
    # counter = 1
    # folderName = os.path.basename(directory_path)
    
    # if folderName == 'x_direction_positive':
    #     initialTheta = 0.0
    # elif folderName == 'x_direction_negative':
    #     initialTheta = np.pi
    # elif folderName == 'y_direction_positive':
    #     initialTheta = np.pi / 2
    # elif folderName == 'y_direction_negative':
    #     initialTheta = -np.pi / 2
    # elif folderName == 'diagonal_first_quad':
    #     initialTheta = np.pi / 4
    # elif folderName == 'diagonal_second_quad':
    #     initialTheta = 3 * np.pi / 4
    # elif folderName == 'diagonal_third_quad':
    #     initialTheta = -3 * np.pi / 4
    # elif folderName == 'diagonal_fourth_quad':
    #     initialTheta = -np.pi / 4

    # for bag_file in bag_files:
    #     # print(bag_file)
    #     print(initialTheta)
    #     process_bag_file(bag_file, counter, initialTheta, folderName)
    #     counter = counter + 1
    
    filenames = ['FullData.csv', 'GT_Deltas.csv', 'Vels_And_Accels.csv']

    combPathsSquare = [
        '../Spezialisierung-1/src/slam_pkg/data/x_direction_positive',
        '../Spezialisierung-1/src/slam_pkg/data/x_direction_negative',
        '../Spezialisierung-1/src/slam_pkg/data/y_direction_positive',
        '../Spezialisierung-1/src/slam_pkg/data/y_direction_negative'
    ]

    combPathsDiagonal_first_and_third_quad = [
        '../Spezialisierung-1/src/slam_pkg/data/diagonal_first_quad',
        '../Spezialisierung-1/src/slam_pkg/data/diagonal_third_quad'
    ]

    combPathsDiagonal_second_and_fourth_quad = [
        '../Spezialisierung-1/src/slam_pkg/data/diagonal_second_quad',
        '../Spezialisierung-1/src/slam_pkg/data/diagonal_fourth_quad'
    ]

    combPathsAll = [
        '../Spezialisierung-1/src/slam_pkg/data/x_direction_positive',
        '../Spezialisierung-1/src/slam_pkg/data/x_direction_negative',
        '../Spezialisierung-1/src/slam_pkg/data/y_direction_positive',
        '../Spezialisierung-1/src/slam_pkg/data/y_direction_negative',
        '../Spezialisierung-1/src/slam_pkg/data/diagonal_first_quad',
        '../Spezialisierung-1/src/slam_pkg/data/diagonal_second_quad',
        '../Spezialisierung-1/src/slam_pkg/data/diagonal_third_quad',
        '../Spezialisierung-1/src/slam_pkg/data/diagonal_fourth_quad'
    ]

    output_dirSquare = '../Spezialisierung-1/src/slam_pkg/data/square'
    output_dirFirstAndThird = '../Spezialisierung-1/src/slam_pkg/data/diagonal_first_and_third_quad'
    output_dirSecondAndFourth = '../Spezialisierung-1/src/slam_pkg/data/diagonal_second_and_fourth_quad'
    output_dirAll = '../Spezialisierung-1/src/slam_pkg/data/AllCombined'

    combine_csv_files(combPathsSquare, filenames, output_dirSquare)
    combine_csv_files(combPathsDiagonal_first_and_third_quad, filenames, output_dirFirstAndThird)
    combine_csv_files(combPathsDiagonal_second_and_fourth_quad, filenames, output_dirSecondAndFourth)
    combine_csv_files(combPathsAll, filenames, output_dirAll)
