#include "RosSubscriber.h"

RosSubscriber::RosSubscriber(ros::NodeHandle n, std::string Topic) : Topic(Topic)
{
    if (this->Topic == "odom") {
        subscriber = n.subscribe<nav_msgs::Odometry>(Topic, 1000, &RosSubscriber::OdomMessageCallback, this);
    }
    
    else if (this->Topic == "scan") {
        subscriber = n.subscribe<sensor_msgs::LaserScan>(Topic, 1000, &RosSubscriber::LaserScanMessageCallback, this);
    }

};


RosSubscriber::~RosSubscriber(){};


void RosSubscriber::OdomMessageCallback(const nav_msgs::Odometry::ConstPtr& OdomMsg)
{
    this->OdomMessage = *OdomMsg;
    std::string OdomToString = boost::lexical_cast<std::string>(this->OdomMessage);
    ROS_INFO("I heard odom: [%s]", OdomToString.c_str());   
};


void RosSubscriber::LaserScanMessageCallback(const sensor_msgs::LaserScan::ConstPtr& LaserMsg)
{
    this->LaserScanMessage = *LaserMsg;
    std::string LaserScanToString = boost::lexical_cast<std::string>(this->LaserScanMessage);
    ROS_INFO("I heard scan: [%s]", LaserScanToString.c_str()); 
}


nav_msgs::Odometry RosSubscriber::getOdom()
{
    return this->OdomMessage;
}


sensor_msgs::LaserScan RosSubscriber::getLaserScan()
{
    return this->LaserScanMessage;
}