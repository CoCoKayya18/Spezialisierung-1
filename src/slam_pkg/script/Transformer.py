#!/usr/bin/env python
import rospy
import tf2_ros
import tf2_geometry_msgs
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped, PoseStamped
import pandas as pd

class WorldFrameTransformer:
    def __init__(self):
        rospy.init_node('world_frame_transformer')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.imu_sub = rospy.Subscriber('/imu', Imu, self.imu_callback)

        self.odom_pub = rospy.Publisher('/odom/world', Odometry, queue_size=10)
        self.imu_pub = rospy.Publisher('/imu/world', Imu, queue_size=10)

        # Initialize Pandas DataFrames
        self.odom_df = pd.DataFrame(columns=['transformation', 'time', 'x', 'y', 'z', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w', 'linearVelocity_x', 'linearVelocity_y', 'linearVelocity_z', 'angularVelocity_x', 'angularVelocity_y', 'angularVelocity_z'])
        self.imu_df = pd.DataFrame(columns=['transformation','time', 'orientation_x', 'orientation_y', 'orientation_z', 'orientation_w', 'angularVelocity_x', 'angularVelocity_y', 'angularVelocity_z', 'linearAcceleration_x', 'linearAcceleration_y', 'linearAcceleration_z'])

        self.odomRows = []
        self.imuRows = []

    def transform_pose(self, pose_msg):
        try:
            transform = self.tf_buffer.lookup_transform('map', pose_msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            transformed_pose = tf2_geometry_msgs.do_transform_pose(pose_msg, transform)
            return transformed_pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr('TF error: {}'.format(e))
            return None

    def transform_imu_orientation(self, imu_msg):
        pose_msg = PoseStamped()
        pose_msg.header = imu_msg.header
        pose_msg.pose.orientation = imu_msg.orientation
        return self.transform_pose(pose_msg)

    def odom_callback(self, msg):
        self.writeToCsv("odom", msg, "noTransform")
        transformed_pose_msg = PoseStamped()
        transformed_pose_msg.header = msg.header
        transformed_pose_msg.pose = msg.pose.pose
        transformed_pose = self.transform_pose(transformed_pose_msg)
        if transformed_pose is not None:
            msg.pose.pose = transformed_pose.pose
            self.writeToCsv("odom", msg, "Transform")
            self.odom_pub.publish(msg)
        
        data = self.odomRows
        self.odom_df = pd.DataFrame(data)
        #self.odom_df.to_csv('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/rosbag_data_2024-03-09-15-26-23/odomTransformed.csv', index=False)

    def imu_callback(self, msg):
        self.writeToCsv("imu", msg, "noTransform")
        transformed_imu = self.transform_imu_orientation(msg)
        if transformed_imu is not None:
            msg.orientation = transformed_imu.pose.orientation
            self.writeToCsv("imu", msg, "Transform")
            self.imu_pub.publish(msg)
        
        data = self.imuRows
        self.imu_df = pd.DataFrame(data)
        #self.imu_df.to_csv('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/rosbag_files/rosbag_data_2024-03-09-15-26-23/imuTransformed.csv', index=False)
    
    def writeToCsv(self, datatype, data, transformed):
        if datatype == "odom":
            # Append a new row to the odom_df DataFrame
            new_row = {
                'transformed' : transformed,
                'time': data.header.stamp.to_sec(),
                'x': data.pose.pose.position.x,
                'y': data.pose.pose.position.y,
                'z': data.pose.pose.position.z,
                'orientation_x': data.pose.pose.orientation.x,
                'orientation_y': data.pose.pose.orientation.y,
                'orientation_z': data.pose.pose.orientation.z,
                'orientation_w': data.pose.pose.orientation.w,
                'linearVelocity_x' : data.twist.twist.linear.x,
                'linearVelocity_y' : data.twist.twist.linear.y,
                'linearVelocity_z' : data.twist.twist.linear.z,
                'angularVelocity_x' : data.twist.twist.angular.x,
                'angularVelocity_y' : data.twist.twist.angular.y,
                'angularVelocity_z' : data.twist.twist.angular.z,
            }
            self.odomRows.append(new_row)
            

        elif datatype == "imu":
            # Append a new row to the imu_df DataFrame
            new_row = {
                'transformed' : transformed,
                'time': data.header.stamp.to_sec(),
                'orientation_x': data.orientation.x,
                'orientation_y': data.orientation.y,
                'orientation_z': data.orientation.z,
                'orientation_w': data.orientation.w,
                'angularVelocity_x' : data.angular_velocity.x,
                'angularVelocity_y' : data.angular_velocity.y,
                'angularVelocity_z' : data.angular_velocity.z,
                'linearAcceleration_x' : data.linear_acceleration.x,
                'linearAcceleration_y' : data.linear_acceleration.y,
                'linearAcceleration_z' : data.linear_acceleration.z,
            }
            
            self.imuRows.append(new_row)

if __name__ == '__main__':
    transformer = WorldFrameTransformer()
    rospy.spin()
