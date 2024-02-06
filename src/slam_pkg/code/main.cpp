#include "EKF_Namespace.h"


using namespace EKF_Namespace;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "subscriber_node");
    ros::NodeHandle nh;

    EKF_Namespace::RosSubscriber odom_sub(nh, "odom");
    EKF_Namespace::RosSubscriber scan_sub(nh, "scan");

    ros::spin();

    EKF_Namespace::EKF_Slam ekf_slam();

    return 0;
}