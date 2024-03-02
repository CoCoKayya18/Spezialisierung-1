#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

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


if __name__ == '__main__':
    rospy.init_node('ros_subscriber_node', anonymous=True)
    OdomSubscriber = RosSubscriber('odom') # Subscribing to 'odom' topic
    GroundTruthSubscriber = RosSubscriber('ground_truth/state')

    Processesor = Process_Input_Data(OdomSubscriber.get_odom(), GroundTruthSubscriber.get_ground_truth())

    rospy.spin()