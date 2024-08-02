#!/usr/bin/env python
import rospy
import actionlib
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
import os
import math

GOAL_FILE_PATH = '../Spezialisierung-1/src/slam_pkg/rosbag_files/random3/goals.txt'
ROTATION_TIMEOUT = 30  # Timeout for rotation in seconds
MAX_RETRIES = 5  # Maximum number of retries for a goal

def move_to_goal(client, x_goal, y_goal, yaw_goal):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'  # Replace with your frame ID
    goal.target_pose.header.stamp = rospy.Time.now()

    goal.target_pose.pose.position.x = x_goal
    goal.target_pose.pose.position.y = y_goal

    q_angle = quaternion_from_euler(0, 0, yaw_goal)
    goal.target_pose.pose.orientation.z = q_angle[2]
    goal.target_pose.pose.orientation.w = q_angle[3]

    client.send_goal(goal)
    wait = client.wait_for_result(rospy.Duration(ROTATION_TIMEOUT))

    if not wait:
        rospy.logerr("Action server not available!")
        return False
    else:
        state = client.get_state()
        if state == actionlib.GoalStatus.SUCCEEDED:
            return True
        else:
            rospy.loginfo("Failed to reach the goal within the timeout.")
            return False

def initialize_action_client():
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to move_base action server.")
    return client

def stop_robot():
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.sleep(1)
    stop_msg = Twist()
    pub.publish(stop_msg)
    rospy.loginfo("Robot stopped.")

def stop_ros_master():
    rospy.loginfo("Stopping ROS master and all nodes.")
    os.system("rosnode kill -a")
    os.system("killall -9 rosmaster")
    os.system("killall -9 roscore")

def save_goal_to_file(x_goal, y_goal, yaw_goal):
    with open(GOAL_FILE_PATH, 'a') as file:
        file.write(f"{x_goal},{y_goal},{yaw_goal}\n")

def follow_path(client, waypoints):
    for idx, waypoint in enumerate(waypoints):
        waypoint_label = f"P{idx + 2}"
        for retry in range(MAX_RETRIES):
            x_goal, y_goal, yaw_goal = waypoint
            save_goal_to_file(x_goal, y_goal, yaw_goal)
            rospy.loginfo("Moving to waypoint %s: x = %s, y = %s meters", waypoint_label, x_goal, y_goal)
            result = move_to_goal(client, x_goal, y_goal, yaw_goal)
            if result:
                rospy.loginfo("Reached waypoint %s successfully.", waypoint_label)
                break
            else:
                rospy.loginfo("Failed to reach waypoint %s. Retrying...", waypoint_label)
                rospy.sleep(1)
        else:
            rospy.loginfo("Failed to reach waypoint %s after maximum retries. Continuing to next waypoint.", waypoint_label)
            continue

def main():
    try:
        rospy.init_node('sendGoal_py')
        client = initialize_action_client()

        # Define your custom snake path here around the columns
        waypoints = [
            #P2
            (1.5, -1.6, math.pi), 
            #P3
            (-0.6, 0.5, math.pi/2), 
            #P4
            (0.7, 1.4, -math.pi/2),
            #P5
            (1.9, 0, -math.pi/4),
            #P6
            (0.5, -2, 2*math.pi/3),
            #P7
            (-0.5, 1.7, 0),
            #P8
            (0.6, 0.7, math.pi),
            #P9
            (-1.8, -0.8, -math.pi/6),
            #P10
            (2, 0.6, -5*math.pi/6),
            #P11
            (0.8, -0.7, math.pi/6),
            #P12
            (0.2, 1.5, -4*math.pi/3),
            #P13
            (0.8, -1.9, 5*math.pi/6),
            #P14
            (-1.8, 0.4, 0),
            #P15
            (1.6, -0.7, math.pi),
            #P16
            (0.5, -2, 2*math.pi/3),
            #P17
            (0.6, 0.7, math.pi),
            #P18
            (0.8, -1.9, 5*math.pi/6),
            #P19
            (2, 0.6, -5*math.pi/6),
            #P20
            (-1.8, 0.4, 0),
            #P22
            (1.5, -1.6, math.pi), 
            #P22
            (0.6, 0.7, math.pi),
            #P23
            (1.9, 0, -math.pi/4),
            #P24
            (-0.5, 1.7, 0),
            #P25
            (0.8, -0.7, math.pi/6),
            #P26
            (-1.8, -0.8, -math.pi/6),
            #P27
            (0.2, 1.5, -4*math.pi/3),
            #P28
            (-0.6, 0.5, math.pi/2), 
            #P29
            (2, 0.6, -5*math.pi/6),
            #P30
            (0.8, -1.9, 5*math.pi/6)
        ]

        follow_path(client, waypoints)

        stop_robot()
        stop_ros_master()

    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation sequence interrupted.")

if __name__ == '__main__':
    main()
