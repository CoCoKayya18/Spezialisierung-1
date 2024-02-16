#include <iostream>
#include "ros/ros.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/LaserScan.h"
#include "std_msgs/String.h"


class RosSubscriber
{
    public:
        RosSubscriber(ros::NodeHandle n, std::string topic);
        ~RosSubscriber();

        void OdomMessageCallback(const nav_msgs::Odometry::ConstPtr& msg);
        void LaserScanMessageCallback(const sensor_msgs::LaserScan::ConstPtr& msg);

        nav_msgs::Odometry getOdom();
        sensor_msgs::LaserScan getLaserScan();

    private:
        std::string Topic;
        ros::Subscriber subscriber;
        nav_msgs::Odometry OdomMessage;
        sensor_msgs::LaserScan LaserScanMessage;
};
