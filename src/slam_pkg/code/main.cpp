#include "EKF_Namespace.h"
#include "RosSubscriber.h"
#include "EKF_Slam_Node.h"


int main(int argc, char **argv)
{
    ros::init(argc, argv, "subscriber_node");
    ros::NodeHandle nh;

    RosSubscriber odom_sub(nh, "odom");
    RosSubscriber scan_sub(nh, "scan");

    ros::spin();

    return 0;
}