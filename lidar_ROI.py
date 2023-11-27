#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

class LidarNode:
    def __init__(self):
        self.node_name = "lidar_node"
        rospy.init_node(self.node_name)

        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.obstacle_pub = rospy.Publisher("obstacle", Bool, queue_size=1)

    def lidar_callback(self, data):
        # 관심 영역 설정 (0도에서 10도 및 350도에서 360도)
        start_angle_1 = 0
        end_angle_1 = 10
        start_angle_2 = 350
        end_angle_2 = 359

        # 라이다 각도별 거리값을 나타내는 배열
        ranges = data.ranges

        # 각도별로 장애물 감지 여부 확인
        obstacle_detected = any([distance <= 0.4 for distance in ranges[start_angle_1:end_angle_1] + ranges[start_angle_2:end_angle_2]])

        rospy.loginfo(f"Obstacle Detected: {obstacle_detected}")
        self.obstacle_pub.publish(obstacle_detected)

if __name__ == '__main__':
    node = LidarNode()
    rospy.spin()
