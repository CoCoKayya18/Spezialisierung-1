#!/usr/bin/env python
import rospy
import actionlib
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
import os

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
        rospy.init_node('move_base_sequence_py')
        client = initialize_action_client()

        goals = [1, 2, 3, 2, 1]  # Distances for each goal in y-direction
        y_current = 0  # Initialize current y position
        x_goal = 0.5  # Constant x goal
        yaw_goal = 1.57079632679  # No rotation

        for i, increment in enumerate(goals):
            y_current += increment  # Update y position by increment
            rospy.loginfo("Sending goal: y = %s meters", y_current)
            result = move_to_goal(client, x_goal, y_current, yaw_goal)
            if result:
                rospy.loginfo("Goal %d reached", i+1)
            else:
                rospy.loginfo("Failed to reach goal %d", i+1)
                break

        if result:
            rospy.loginfo("All goals reached successfully. Shutting down.")
            stop_robot()  # Ensure the robot is stopped
            stop_ros_master()  # Shutdown ROS master

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation sequence interrupted.")

if __name__ == '__main__':
    main()
