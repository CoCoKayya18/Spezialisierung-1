#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R

class RosSubscriber:

    def __init__(self, topic):
        
        self.topic = topic
        
        if self.topic == "odom":
            self.subscriber = rospy.Subscriber(topic, Odometry, self.odom_message_callback)
        elif self.topic == "scan":
            self.subscriber = rospy.Subscriber(topic, LaserScan, self.laser_scan_message_callback)
        elif self.topic == "ground_truth/state":
            self.subscriber = rospy.Subscriber(topic, Odometry, self.ground_truth_message_callback)
        
        # Initialize message storage
        self.odom_message = None
        self.laser_scan_message = None
        self.ground_truth_message = None

    def odom_message_callback(self, msg):
        self.odom_message = msg
        rospy.loginfo("I heard odom: %s", str(msg))

    def laser_scan_message_callback(self, msg):
        self.laser_scan_message = msg
        rospy.loginfo("I heard scan: %s", str(msg))

    def ground_truth_message_callback(self, msg):
        self.ground_truth_message = msg
        rospy.loginfo("I heard ground_truth: %s", str(msg))

    def get_odom(self):
        return self.odom_message

    def get_laser_scan(self):
        return self.laser_scan_message
    
    def get_ground_truth(self):
        return self.ground_truth_message

class Process_Input_Data:
    def __init__(self, input_odomData, inputGroundTruthData):
        odom_data = input_odomData
        ground_Truth_Data = inputGroundTruthData
        
        deltaX = None
        ground_Truth_Data = None
        prior_groundTruth = None
    
    def save_as_prior(self, currentGroundTruth):
        prior_groundTruth = currentGroundTruth

    def writeToCSV(self):
        testing=0

    def calculate_deltaX(self):
        self.deltaX.positionX = self.ground_Truth_Data.pose.pose.position.x - self.prior_groundTruth.pose.pose.position.x
        self.deltaX.positionY = self.ground_Truth_Data.pose.pose.position.y - self.prior_groundTruth.pose.pose.position.y

        prior_roll, prior_pitch, prior_yaw = euler_from_quaternion(self.prior_groundTruth)
        current_roll, current_pitch, current_yaw = euler_from_quaternion(self.ground_Truth_Data)

        self.deltaX.positionTheta = current_yaw - prior_yaw



# Helper functions for transformations and other operations
# ... (to be implemented as needed) ...

class EKFSLAM:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.mu = initial_state  # State vector
        self.Sigma = initial_covariance  # Covariance matrix
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance
        self.landmarks = {}  # Dictionary to store landmarks

    def predict(self, u):
        # Prediction step with a Gaussian process
        # Here u is the control input, which could be odometry or velocity commands
        # ... (implement prediction equations) ...
        pass

    def update(self, measurements):
        # Update step with the obtained measurements
        # ... (implement update equations) ...
        pass

    def extract_features(self, point_cloud):
        # Feature extraction from 2D Lidar data
        # ... (implement downsampling, LOAM filtering, etc.) ...
        return extracted_features

    def match_landmarks(self, features):
        # Landmark matching and updating the self.landmarks dictionary
        # ... (implement matching algorithm) ...
        pass

    def run(self, control_input, point_cloud):
        # Main EKF SLAM loop iteration
        self.predict(control_input)
        features = self.extract_features(point_cloud)
        self.match_landmarks(features)
        self.update(features)

# # Sample usage
# # Define initial state, covariance, and noises
# initial_state = np.zeros((3,))
# initial_covariance = np.eye(3)
# process_noise = np.diag([0.1, 0.1, np.deg2rad(5)])  # example values
# measurement_noise = np.diag([0.1, 0.1, np.deg2rad(1)])  # example values

# # Initialize EKF SLAM
# ekf_slam = EKFSLAM(initial_state, initial_covariance, process_noise, measurement_noise)

# # Assume you have control inputs and Lidar scans over time
# for control_input, point_cloud in zip(control_inputs, lidar_scans):
#     ekf_slam.run(control_input, point_cloud)



if __name__ == '__main__':
    rospy.init_node('ros_subscriber_node', anonymous=True)
    OdomSubscriber = RosSubscriber('odom') # Subscribing to 'odom' topic
    GroundTruthSubscriber = RosSubscriber('ground_truth/state')

    Processesor = Process_Input_Data(OdomSubscriber.get_odom(), GroundTruthSubscriber.get_ground_truth())

    rospy.spin()