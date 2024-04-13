#!/usr/bin/env python
import rospy
import actionlib
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
import os

def move_to_goal(x_goal, y_goal, yaw_goal):
    # Create an action client called "move_base" with action definition file "MoveBaseAction"
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    # Waits until the action server has started up and started listening for goals.
    client.wait_for_server()

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
    else:
        # Result of executing the action
        return client.get_result()

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

# Initializes a rospy node to let the SimpleActionClient publish and subscribe
if __name__ == '__main__':
    try:
        rospy.init_node('move_base_goal_py')
        x_goal = 3.0  # Change to your x goal
        y_goal = 0.5  # Change to your y goal
        yaw_goal = 0  # Change to your yaw goal (in radians)

        result = move_to_goal(x_goal, y_goal, yaw_goal)
        if result:
            rospy.loginfo("Goal execution done!")
            stop_robot()  # Stop the robot when the goal is reached
            stop_ros_master()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
