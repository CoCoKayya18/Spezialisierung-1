from bagpy import bagreader
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os


class bagReader:

    def __init__(self, bagfilepath):
        self.bag = bagreader(bagfilepath)

    def extract_topics(self):
        odom_csv = self.bag.message_by_topic('odom/world')
        ground_truth_csv = self.bag.message_by_topic('ground_truth/state')
        imu_csv = self.bag.message_by_topic('imu/world')

    def process_data(odom_df, imu_df, ground_truth_df):
        return

class BagDataProcessor:

    def __init__(self, bag_file_path):
        self.bag = bagreader(bag_file_path)

    def read_topic(self, topic_name):
        csv_path = self.bag.message_by_topic(topic_name)
        if csv_path is not None:
            return pd.read_csv(csv_path)
        else:
            return None
        
    def getAccelAngularTheta(self, imu_df):

        imu_df['Time'] = pd.to_numeric(imu_df['Time'])
        imu_df['angular_velocity.z'] = pd.to_numeric(imu_df['angular_velocity.z'])

        angular_accelerations = [0]  # Initialize with 0 for the first record
        times = imu_df['Time'].values
        angular_velocities = imu_df['angular_velocity.z'].values

        for i in range(1, len(imu_df)):
            delta_time = times[i] - times[i - 1]
            delta_angular_velocity = angular_velocities[i] - angular_velocities[i - 1]
            angular_acceleration = delta_angular_velocity / delta_time
            angular_accelerations.append(angular_acceleration)
        
        imu_df['angular_acceleration.theta'] = angular_accelerations
        return imu_df

    def calculate_deltaX(self, ground_truth_df):
        
        delta_x, delta_y, delta_z, delta_yaw = [0], [0], [0], [0]

        for i in range(1, len(ground_truth_df)):
        # Calculate the deltas for position
            delta_x.append(ground_truth_df.iloc[i]['pose.pose.position.x'] - ground_truth_df.iloc[i-1]['pose.pose.position.x'])
            delta_y.append(ground_truth_df.iloc[i]['pose.pose.position.y'] - ground_truth_df.iloc[i-1]['pose.pose.position.y'])
            delta_z.append(ground_truth_df.iloc[i]['pose.pose.position.z'] - ground_truth_df.iloc[i-1]['pose.pose.position.z'])
            
            # Calculate the delta for rotation around the z-axis
            current_orientation = [ground_truth_df.iloc[i][col] for col in ['pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w']]
            previous_orientation = [ground_truth_df.iloc[i-1][col] for col in ['pose.pose.orientation.x', 'pose.pose.orientation.y', 'pose.pose.orientation.z', 'pose.pose.orientation.w']]
            
            # Convert quaternions to Euler angles
            current_euler = R.from_quat(current_orientation).as_euler('xyz', degrees=True)
            previous_euler = R.from_quat(previous_orientation).as_euler('xyz', degrees=True)
            
            # Append the change in yaw (z-axis rotation)
            delta_yaw.append(current_euler[2] - previous_euler[2])

        # Add the calculated delta positions back to the dataframe
        ground_truth_df['delta_x'] = delta_x
        ground_truth_df['delta_y'] = delta_y
        ground_truth_df['delta_z'] = delta_z
        ground_truth_df['delta_yaw'] = delta_yaw
        return ground_truth_df
    
    def find_closes_timestamp(self, ground_truth_df, imu_df, odom_df):
        mergeGTandImuDf = pd.merge_asof(ground_truth_df, imu_df, on='Time')
        mergeGTandIMUandOdomDf = pd.merge_asof(mergeGTandImuDf, odom_df, on='Time')
        mergeGTandIMUandOdomDf.to_csv('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/FullDatasetRosbag/mergedData.csv')
        return mergeGTandIMUandOdomDf

        
    def process_data(self, odom_df, imu_df, ground_truth_df):

        ground_truth_df = self.calculate_deltaX(ground_truth_df)
        imu_df = self.getAccelAngularTheta(imu_df)

        MergedDf = self.find_closes_timestamp(ground_truth_df, imu_df, odom_df)

        data = []
        
        for index, row in MergedDf.iterrows():

            GT_X = row['pose.pose.position.x_x']
            GT_Y = row['pose.pose.position.y_x']
            GT_Quaternions = [row['pose.pose.orientation.x_x'], row['pose.pose.orientation.y_x'], row['pose.pose.orientation.z_x'], row['pose.pose.orientation.w_x']]
            GT_Euler = R.from_quat(GT_Quaternions).as_euler('xyz', degrees=True)
            GTYaw = GT_Euler[2]

            VelLinearX = row['twist.twist.linear.x_x']
            VelLinearY = row['twist.twist.linear.y_x']
            # VelLinearZ = row['twist.twist.linear.z_x']

            # VelAngularX = row['twist.twist.angular.x_x']
            # VelAngularY = row['twist.twist.angular.y_x']
            VelAngularYaw = row['twist.twist.angular.z_x']

            AccelLinearX = row['linear_acceleration.x']
            AccelLinearY = row['linear_acceleration.y']
            AccelAngularYaw = row['angular_acceleration.theta']

            deltaX_X = row['delta_x']
            deltaX_Y = row['delta_y']
            #deltaX_Z = row['delta_z']
            deltaX_Yaw = row['delta_yaw']

            # row = [VelLinearX, VelLinearY, VelLinearZ, VelAngularX, VelAngularY, VelAngularZ, AccelLinearX, AccelLinearY, AccelAngularTheta, deltaX_X, deltaX_Y, deltaX_Yaw]
            row = [GT_X, GT_Y, GTYaw, VelLinearX, VelLinearY, VelAngularYaw, AccelLinearX, AccelLinearY, AccelAngularYaw, deltaX_X, deltaX_Y, deltaX_Yaw]
            data.append(row)

        datapointsForGP = pd.DataFrame(data)
        #datapointsForGP.columns = ['Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Linear_Z', 'Velocity_Angular_X', 'Velocity_Angular_Y', 'Velocity_Angular_Z', 'Accel_Linear_X', 'Accel_Linear_Y', 'Accel_Linear_Z', 'Accel_Angular_Theta' 'Delta_X_X', 'Delta_X_Y', 'delta_X_Theta']
        datapointsForGP.columns = ['GroundTruth_X', 'Ground_Truth_Y', 'Ground_Truth_Yaw', 'Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Angular_Yaw', 'Accel_Linear_X', 'Accel_Linear_Y', 'Accel_Angular_Yaw', 'Delta_X_X', 'Delta_X_Y', 'delta_X_Yaw']

        file_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'

        if os.path.getsize(file_path) > 0:
            datapointsForGP.to_csv(file_path, mode='a', index=False, header=False)
        
        else:
            datapointsForGP.to_csv(file_path, mode='a', index=False)    





    def save_to_csv(self, df, file_name):
        df.to_csv(file_name, index=False)
    

if __name__ == '__main__':

    filepath = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/FullDatasetRosbag.bag'
    processor = BagDataProcessor(filepath)

    odom_df = processor.read_topic('odom/world')
    imu_df = processor.read_topic('imu/world')
    ground_truth_df = processor.read_topic('ground_truth/state')

    # Process the data to calculate deltas, velocities, and accelerations
    processor.process_data(odom_df, imu_df, ground_truth_df)
