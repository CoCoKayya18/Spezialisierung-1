#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
import pandas as panda
import os
import datetime

class RosSubscriber:

    def __init__(self, topic):
        
        self.topic = topic
        
        if self.topic == "odom":
            self.subscriber = rospy.Subscriber(topic, Odometry, self.odom_message_callback)
        elif self.topic == "ground_truth/state":
            self.subscriber = rospy.Subscriber(topic, Odometry, self.ground_truth_message_callback)
        
        # Initialize message storage
        self.odom_message = None
        self.ground_truth_message = None

    def odom_message_callback(self, msg):
        self.odom_message = msg
        # rospy.loginfo("I heard odom: %s", str(msg))

    def ground_truth_message_callback(self, msg):
        self.ground_truth_message = msg
        # rospy.loginfo("I heard ground_truth: %s", str(msg))

    def get_odom(self):
        return self.odom_message
    
    def get_ground_truth(self):
        return self.ground_truth_message

class Process_Input_Data:

    def __init__(self):

        self.odom_data = Odometry()
        self.ground_Truth_Data = Odometry()
        
        self.deltaX = [0,0,0]
        self.prior_groundTruth = None
        rospy.loginfo("INITIALISATION")

    
    def save_as_prior(self, currentGroundTruth):
        if self.prior_groundTruth is None:
            self.prior_groundTruth = currentGroundTruth
            rospy.loginfo("PRIOR SET")

    def calculate_deltaX(self, odomData, groundTruthData):

        self.odom_data = odomData
        self.ground_Truth_Data = groundTruthData

        self.save_as_prior(self.ground_Truth_Data)

        self.deltaX[0] = self.ground_Truth_Data.pose.pose.position.x - self.prior_groundTruth.pose.pose.position.x
        self.deltaX[1] = self.ground_Truth_Data.pose.pose.position.y - self.prior_groundTruth.pose.pose.position.y

        self.prior_Quaternions = [self.prior_groundTruth.pose.pose.orientation.x, self.prior_groundTruth.pose.pose.orientation.y, self.prior_groundTruth.pose.pose.orientation.z, self.prior_groundTruth.pose.pose.orientation.w]
        self.current_Quaternions = [self.ground_Truth_Data.pose.pose.orientation.x, self.ground_Truth_Data.pose.pose.orientation.y, self.ground_Truth_Data.pose.pose.orientation.z, self.ground_Truth_Data.pose.pose.orientation.w]

        prior_roll, prior_pitch, prior_yaw = euler_from_quaternion(self.prior_Quaternions)
        current_roll, current_pitch, current_yaw = euler_from_quaternion(self.current_Quaternions)

        self.deltaX[2] = current_yaw - prior_yaw

        # rospy.loginfo("Odom timestamp: %s", datetime.datetime.fromtimestamp(self.odom_data.header.stamp.secs + self.odom_data.header.stamp.nsecs/1e9))
        # rospy.loginfo("GroundTruth timestamp: %s", datetime.datetime.fromtimestamp(self.ground_Truth_Data.header.stamp.secs + self.ground_Truth_Data.header.stamp.nsecs/1e9))

        rospy.loginfo("I heard deltaX: %s", str(self.deltaX))

        self.writeToCSV(self.odom_data, self.deltaX)

        self.prior_groundTruth = self.ground_Truth_Data

    def writeToCSV(self, writeOdomData, writeDeltaX):

        odomPosition = [writeOdomData.pose.pose.position.x, writeOdomData.pose.pose.position.y, writeOdomData.pose.pose.position.z]
        odomVelocity = [writeOdomData.twist.twist.linear.x, writeOdomData.twist.twist.linear.y, writeOdomData.twist.twist.angular.z]
        

        data = [odomPosition, odomVelocity, writeDeltaX]

        df = panda.DataFrame(data)
        df.columns = ['Odometry_Position', 'Odometry_Velocity', 'Delta_X']

        file_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'
        
        if os.path.getsize(file_path) > 0:
            df.to_csv(file_path, mode='a', index=False, header=False)
        
        else:
            rospy.loginfo("Header created")
            df.to_csv(file_path, mode='a', index=False)

# Main Loop

if __name__ == '__main__':

    rospy.init_node('ros_subscriber_node', anonymous=True)
    OdomSubscriber = RosSubscriber('odom')
    GroundTruthSubscriber = RosSubscriber('ground_truth/state')

    # Wait for the first message to be published

    while not rospy.is_shutdown() and (OdomSubscriber.get_odom() is None or GroundTruthSubscriber.get_ground_truth() is None):
        rospy.sleep(0.1)
    
    # Set the rate to the same as odom and ground truth are beeing published
    rate = rospy.Rate(30)

    Processesor = Process_Input_Data()

    # Loop the Dataprocessing to receive as many data as possible
    while not rospy.is_shutdown():
        try:
            Processesor.calculate_deltaX(OdomSubscriber.get_odom(), GroundTruthSubscriber.get_ground_truth())

        except rospy.ROSInterruptException:
            break

        rate.sleep()