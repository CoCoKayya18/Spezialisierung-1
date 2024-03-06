#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
import pandas as panda
import os
import datetime
import message_filters

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
    
    def get_timestamp(self):
        return self.timestamp

class Process_Input_Data:

    def __init__(self):

        self.odom_data = Odometry()
        self.ground_Truth_Data = Odometry()
        
        self.deltaX = [0,0,0]
        self.prior_groundTruth = None
        #rospy.loginfo("INITIALISATION")

    
    def save_as_prior(self, currentGroundTruth):
        if self.prior_groundTruth is None:
            self.prior_groundTruth = currentGroundTruth
            #rospy.loginfo("PRIOR SET")

    def calculate_deltaX(self, odomData, groundTruthData):

        self.odom_data = odomData
        self.ground_Truth_Data = groundTruthData

        self.save_as_prior(self.ground_Truth_Data)

        if self.prior_groundTruth == self.ground_Truth_Data:
            rospy.loginfo("Prior set")
            return
        
        else:
            rospy.loginfo("Prior already set, calculating data")
            self.deltaX[0] = self.ground_Truth_Data.pose.pose.position.x - self.prior_groundTruth.pose.pose.position.x
            self.deltaX[1] = self.ground_Truth_Data.pose.pose.position.y - self.prior_groundTruth.pose.pose.position.y

            self.prior_Quaternions = [self.prior_groundTruth.pose.pose.orientation.x, self.prior_groundTruth.pose.pose.orientation.y, self.prior_groundTruth.pose.pose.orientation.z, self.prior_groundTruth.pose.pose.orientation.w]
            self.current_Quaternions = [self.ground_Truth_Data.pose.pose.orientation.x, self.ground_Truth_Data.pose.pose.orientation.y, self.ground_Truth_Data.pose.pose.orientation.z, self.ground_Truth_Data.pose.pose.orientation.w]

            prior_roll, prior_pitch, prior_yaw = euler_from_quaternion(self.prior_Quaternions)
            current_roll, current_pitch, current_yaw = euler_from_quaternion(self.current_Quaternions)

            self.deltaX[2] = current_yaw - prior_yaw

            rospy.loginfo("I heard deltaX: %s", str(self.deltaX))

            self.writeToCSV(self.odom_data, self.deltaX)

            self.prior_groundTruth = self.ground_Truth_Data

    def writeToCSV(self, writeOdomData, writeDeltaX):

        odomPositionX = writeOdomData.pose.pose.position.x
        odomPositionY = writeOdomData.pose.pose.position.y
        odomPositionZ = writeOdomData.pose.pose.position.z

        odomVelocityX = writeOdomData.twist.twist.linear.x
        odomVelocityY = writeOdomData.twist.twist.linear.y
        odomVelocityZ = writeOdomData.twist.twist.angular.z

        deltaX_X = writeDeltaX[0]
        deltaX_Y = writeDeltaX[1]
        deltaX_Z = writeDeltaX[2]

        data = [[odomPositionX, odomPositionY, odomPositionZ, odomVelocityX, odomVelocityY, odomVelocityZ, deltaX_X, deltaX_Y, deltaX_Z]]

        df = panda.DataFrame(data)
        df.columns = ['Odometry_Position_X', 'Odometry_Position_Y', 'Odometry_Position_Z', 'Odometry_Velocity_X', 'Odometry_Velocity_Y', 'Odometry_Velocity_Z', 'Delta_X_X', 'Delta_X_Y', 'Delta_X_Z']

        file_path = '/home/cocokayya18/Spezialisierung-1/src/slam_pkg/data/Data.csv'
        
        if os.path.getsize(file_path) > 0:
            df.to_csv(file_path, mode='a', index=False, header=False)
        
        else:
            rospy.loginfo("Header created")
            df.to_csv(file_path, mode='a', index=False)


class filterSubscriber:

    def __init__(self):
        
        # Subscribers using message_filters
        odom_sub = message_filters.Subscriber('odom', Odometry)
        ground_truth_sub = message_filters.Subscriber('ground_truth/state', Odometry)

        # ApproximateTime Synchronizer
        ats = message_filters.ApproximateTimeSynchronizer([odom_sub, ground_truth_sub], queue_size=10, slop=0.05)
        ats.registerCallback(self.filterCallback)

        self.odometry_msg = None
        self.ground_truth_msg = None

    def filterCallback(self, odometry_msg, ground_truth_msg):

        self.odometry_msg = odometry_msg
        self.ground_truth_msg = ground_truth_msg

    def get_odom_msg(self):
        return self.odometry_msg

    def get_ground_truth_msg(self):
        return self.ground_truth_msg

# Main Loop

if __name__ == '__main__':

    rospy.init_node('ros_subscriber_node', anonymous=True)

    rate = rospy.Rate(30)

    odom_msg_waiter = rospy.wait_for_message('odom', Odometry, timeout=None)
    ground_truth_msg_waiter = rospy.wait_for_message('ground_truth/state', Odometry, timeout=None)


    filterSub = filterSubscriber()

    processor = Process_Input_Data()
    # rospy.loginfo("Here yuzi")
    
    while not rospy.is_shutdown():
        try:
            processor.calculate_deltaX(filterSub.get_odom_msg(), filterSub.get_ground_truth_msg())
        except rospy.ROSInterruptException:
            break
        rate.sleep()