#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
import pandas as panda
import os
import message_filters
import math

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

        self.odom_Data = Twist()
        self.ground_Truth_Data = Odometry()
        
        self.deltaX = [0,0,0]
        self.prior_groundTruth = None
        #rospy.loginfo("INITIALISATION")

    
    def save_as_prior(self, currentGroundTruth):
        if self.prior_groundTruth is None:
            self.prior_groundTruth = currentGroundTruth
            self.prior_timestamp = self.imu_Data.header.stamp.to_sec()
            #rospy.loginfo("PRIOR SET")

    # def groundTruth_StateChange(self, priorGT, currentGT, tolerance):
    #     position_changed = False
    #     orientation_changed = False

    #     # Calculate position difference
    #     pos_diff = math.sqrt((currentGT.pose.pose.position.x - priorGT.pose.pose.position.x) ** 2 +
    #                     (currentGT.pose.pose.position.y - priorGT.pose.pose.position.y) ** 2 +
    #                     (currentGT.pose.pose.position.z - priorGT.pose.pose.position.z) ** 2)

    #     if pos_diff > tolerance:
    #         position_changed = True

    #     # Calculate orientation difference using quaternion distance
    #     current_orientation = [currentGT.pose.pose.orientation.x,
    #                         currentGT.pose.pose.orientation.y,
    #                         currentGT.pose.pose.orientation.z,
    #                         currentGT.pose.pose.orientation.w]

    #     previous_orientation = [priorGT.pose.pose.orientation.x,
    #                             priorGT.pose.pose.orientation.y,
    #                             priorGT.pose.pose.orientation.z,
    #                             priorGT.pose.pose.orientation.w]

    #     # Dot product between the two quaternions
    #     dot_product = sum(c * p for c, p in zip(current_orientation, previous_orientation))
        
    #     # Calculate angle difference; clamp dot product to valid range for acos
    #     angle_diff = math.acos(max(min(dot_product, 1.0), -1.0)) * 2
        
    #     if angle_diff > tolerance:
    #         orientation_changed = True

    #     return position_changed or orientation_changed
            
    def calculate_angularAccel(self, imuData, timediff):
        angular_acceleration_z = (imuData.angular_velocity.z - self.previous_angular_velocity.z) / timediff
        return angular_acceleration_z

    def calculate_deltaX(self, odomData, groundTruthData, imuData):

        self.odom_Data = odomData
        self.ground_Truth_Data = groundTruthData
        self.imu_Data = imuData
        self.prior_timestamp = None

        if imuData is None:
            rospy.loginfo("IMU data is None. Skipping this cycle.")
            return

        self.save_as_prior(self.ground_Truth_Data)

        # change = self.groundTruth_StateChange(self, self.prior_groundTruth, self.ground_Truth_Data, 0.001)

        change = True
        
        if self.ground_Truth_Data == None:
            rospy.loginfo("Position didnt change")
            return
        
        else:
            rospy.loginfo("Calculating deltaX")
            self.deltaX[0] = self.ground_Truth_Data.pose.pose.position.x - self.prior_groundTruth.pose.pose.position.x
            self.deltaX[1] = self.ground_Truth_Data.pose.pose.position.y - self.prior_groundTruth.pose.pose.position.y

            self.prior_Quaternions = [self.prior_groundTruth.pose.pose.orientation.x, self.prior_groundTruth.pose.pose.orientation.y, self.prior_groundTruth.pose.pose.orientation.z, self.prior_groundTruth.pose.pose.orientation.w]
            self.current_Quaternions = [self.ground_Truth_Data.pose.pose.orientation.x, self.ground_Truth_Data.pose.pose.orientation.y, self.ground_Truth_Data.pose.pose.orientation.z, self.ground_Truth_Data.pose.pose.orientation.w]

            prior_roll, prior_pitch, prior_yaw = euler_from_quaternion(self.prior_Quaternions)
            current_roll, current_pitch, current_yaw = euler_from_quaternion(self.current_Quaternions)

            self.deltaX[2] = current_yaw - prior_yaw

            rospy.loginfo("I heard deltaX: %s", str(self.deltaX))

            self.writeToCSV(self.odom_Data, self.imu_Data, self.deltaX)

            self.prior_groundTruth = self.ground_Truth_Data
            self.prior_timestamp = self.imu_Data.header.stamp.to_sec()

    def writeToCSV(self, writeodom, writeImu, writeDeltaX):

        VelLinearX = writeodom.twist.twist.linear.x
        VelLinearY = writeodom.twist.twist.linear.y
        VelLinearZ = writeodom.twist.twist.linear.z

        VelAngularX = writeodom.twist.twist.angular.x
        VelAngularY = writeodom.twist.twist.angular.y
        VelAngularZ = writeodom.twist.twist.angular.z

        AccelLinearX = writeImu.linear_acceleration.x
        AccelLinearY = writeImu.linear_acceleration.y

        current_time = self.imu_Data.header.stamp.to_sec()
        
        if self.prior_timestamp is not None:
            dt = current_time - self.prior_timestamp
        
        else:
            dt = 0

        if dt>0:
            AccelAngularZ = self.calculate_angularAccel(self.imu_Data, dt)
        else:
            AccelAngularZ = 0


        deltaX_X = writeDeltaX[0]
        deltaX_Y = writeDeltaX[1]
        deltaX_Theta = writeDeltaX[2]

        data = [[VelLinearX, VelLinearY, VelLinearZ, VelAngularX, VelAngularY, VelAngularZ, AccelLinearX, AccelLinearY, AccelAngularZ, deltaX_X, deltaX_Y, deltaX_Theta]]

        df = panda.DataFrame(data)
        df.columns = ['Velocity_Linear_X', 'Velocity_Linear_Y', 'Velocity_Linear_Z', 'Velocity_Angular_X', 'Velocity_Angular_Y', 'Velocity_Angular_Z', 'Accel_Linear_X', 'Accel_Linear_Y', 'Accel_Linear_Z', 'Accel_Angular_Theta' 'Delta_X_X', 'Delta_X_Y', 'delta_X_Theta']

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
        imu = message_filters.Subscriber('imu', Imu)

        # ApproximateTime Synchronizer
        ats = message_filters.ApproximateTimeSynchronizer([odom_sub, ground_truth_sub, imu], queue_size=50, slop=0.011, allow_headerless=True)
        ats.registerCallback(self.filterCallback)

        self.odom_msg = None
        self.ground_truth_msg = None
        self.imu_msg = None

    def filterCallback(self, odom_msg, ground_truth_msg, imu_msg):

        self.odom_msg = odom_msg
        self.ground_truth_msg = ground_truth_msg
        self.imu_msg = imu_msg

    def get_odom_msg(self):
        return self.odom_msg

    def get_ground_truth_msg(self):
        return self.ground_truth_msg
    
    def get_imu_msg(self):
        return self.imu_msg

# Main Loop

if __name__ == '__main__':

    rospy.init_node('ros_subscriber_node', anonymous=True)

    rate = rospy.Rate(30)

    odom_msg_waiter = rospy.wait_for_message('odom', Odometry, timeout=None)
    ground_truth_msg_waiter = rospy.wait_for_message('ground_truth/state', Odometry, timeout=None)
    imu_msg_waiter = rospy.wait_for_message('imu', Imu, timeout=None)

    filterSub = filterSubscriber()

    processor = Process_Input_Data()
    
    while not rospy.is_shutdown():
        try:
            processor.calculate_deltaX(filterSub.get_odom_msg(), filterSub.get_ground_truth_msg(), filterSub.get_imu_msg())
        except rospy.ROSInterruptException:
            break
        rate.sleep()