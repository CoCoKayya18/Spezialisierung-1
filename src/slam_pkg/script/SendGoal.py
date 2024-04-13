#!/usr/bin/env python
import rospy
import actionlib
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
import os
import math

def move_to_goal(client, x_goal, y_goal, yaw_goal):
    # Creates a new goal with the MoveBaseGoal constructor
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'  # Replace with your frame ID
    goal.target_pose.header.stamp = rospy.Time.now()

    # Set goal position
    goal.target_pose.pose.position.x = x_goal
    goal.target_pose.pose.position.y = y_goal

    # Convert yaw to quaternion
    q_angle = quaternion_from_euler(0, 0, yaw_goal)
    goal.target_pose.pose.orientation.z = q_angle[2]
    goal.target_pose.pose.orientation.w = q_angle[3]

    # Sends the goal to the action server.
    client.send_goal(goal)

    # Waits for the server to finish performing the action.
    wait = client.wait_for_result()

    # If the result doesn't arrive, assume the Server is not available
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
        return False
    else:
        # Result of executing the action
        return client.get_result()

def initialize_action_client():
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()
    return client

def stop_robot():
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.sleep(1)  # Give some time for the publisher to register in the ROS network
    stop_msg = Twist()  # Empty Twist message has linear.x = 0 and angular.z = 0 by default
    pub.publish(stop_msg)
    rospy.loginfo("Robot stopped.")

def stop_ros_master():
    rospy.loginfo("Stopping ROS master and all nodes.")
    os.system("rosnode kill -a")  # Kills all nodes
    os.system("killall -9 rosmaster")  # Stops the ROS master
    os.system("killall -9 roscore")  # Stops the roscore as well

def main():
    try:
        rospy.init_node('move_base_diagonal_py')
        client = initialize_action_client()

        # Diagonal goal to point (4,4)
        x_goal = -4.0
        y_goal = -4.0
        yaw_goal = -math.pi / 4  # 45 degrees, adjust this angle as needed

        rospy.loginfo("Sending diagonal goal: x = %s, y = %s meters", x_goal, y_goal)
        result = move_to_goal(client, x_goal, y_goal, yaw_goal)
        if result:
            rospy.loginfo("Diagonal goal reached successfully.")
        else:
            rospy.loginfo("Failed to reach the diagonal goal")

        stop_robot()  # Ensure the robot is stopped after reaching the goal or failing
        stop_ros_master()  # Shutdown ROS master

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation sequence interrupted.")

if __name__ == '__main__':
    main()

### Diagonal Movement ###
