#!/usr/bin/env python

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid
from random import uniform
import numpy as np

class MapAwareGoalGenerator:
    def __init__(self):
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.map_subscriber = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.map_data = None
        rospy.loginfo("Waiting for move_base action server...")
        self.client.wait_for_server()
        rospy.loginfo("Connected to move_base server")

    def map_callback(self, data):
        self.map_data = data

    def send_random_goal(self):
        if not self.map_data:
            rospy.loginfo("Map data not yet received.")
            return
        
        goal_sent = False
        while not goal_sent:
            x_goal, y_goal = self.generate_random_goal()
            if self.is_goal_valid(x_goal, y_goal):
                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "map"
                goal.target_pose.header.stamp = rospy.Time.now()
                goal.target_pose.pose.position.x = x_goal
                goal.target_pose.pose.position.y = y_goal
                goal.target_pose.pose.orientation.w = 1.0
                self.client.send_goal(goal)
                rospy.loginfo(f"Sending new random valid goal: x={x_goal}, y={y_goal}")
                goal_sent = True
            else:
                rospy.loginfo("Generated goal is invalid, generating a new one...")
        self.client.wait_for_result()
        rospy.loginfo("Goal Reached!")

    def generate_random_goal(self):
        # Generate a random goal within the map bounds
        width = self.map_data.info.width * self.map_data.info.resolution
        height = self.map_data.info.height * self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y

        x_goal = uniform(origin_x, origin_x + width)
        y_goal = uniform(origin_y, origin_y + height)

        return x_goal, y_goal

    def is_goal_valid(self, x, y):
        # Transform from world coordinates to map coordinates
        x_index = int((x - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        y_index = int((y - self.map_data.info.origin.position.y) / self.map_data.info.resolution)

        if 0 <= x_index < self.map_data.info.width and 0 <= y_index < self.map_data.info.height:
            index = x_index + y_index * self.map_data.info.width
            # Check if the point is not in an obstacle
            return self.map_data.data[index] < 50  # Free space threshold
        return False

if __name__ == '__main__':
    try:
        rospy.init_node('random_goal_generator')
        explorer = MapAwareGoalGenerator()
        rate = rospy.Rate(0.05)  # Adjust based on your needs
        while not rospy.is_shutdown():
            explorer.send_random_goal()
            rate.sleep()
    except rospy.ROSInterruptException:
        rospy.loginfo("Random Goal Generator node terminated.")
