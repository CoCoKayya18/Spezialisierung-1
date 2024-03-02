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
    def __init__(self, input_odomData, inputGroundTruthData):
        self.odom_data = input_odomData
        self.ground_Truth_Data = inputGroundTruthData
        
        self.deltaX = [0,0,0]
        # self.ground_Truth_Data = None
        self.prior_groundTruth = Odometry 

        self.calculate_deltaX()
    
    def save_as_prior(self, currentGroundTruth):
        prior_groundTruth = currentGroundTruth

    def writeToCSV(self):
        testing=0

    def calculate_deltaX(self):
        self.deltaX[0] = self.ground_Truth_Data.pose.pose.position.x - self.prior_groundTruth.pose.pose.position.x
        self.deltaX[1] = self.ground_Truth_Data.pose.pose.position.y - self.prior_groundTruth.pose.pose.position.y

        prior_roll, prior_pitch, prior_yaw = euler_from_quaternion(self.prior_groundTruth)
        current_roll, current_pitch, current_yaw = euler_from_quaternion(self.ground_Truth_Data)

        self.deltaX[2] = current_yaw - prior_yaw

        rospy.loginfo("I heard deltaX: %s", str(self.deltaX))


if __name__ == '__main__':
    rospy.init_node('ros_subscriber_node', anonymous=True)
    OdomSubscriber = RosSubscriber('odom') # Subscribing to 'odom' topic
    GroundTruthSubscriber = RosSubscriber('ground_truth/state')

    Processesor = Process_Input_Data(OdomSubscriber.get_odom(), GroundTruthSubscriber.get_ground_truth())



    rospy.spin()