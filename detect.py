#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt


class LaneDetectionNode:
    def __init__(self):
        self.node_name = "birdseye_lane_detection_node"
        rospy.init_node(self.node_name)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/lane_detection/birdseye_image", Image, queue_size=1)


    def image_callback(self, img_msg):
      cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
      hsvResult=self.hsvExtraction(cv_image)
      bird_eye_view = self.birdseye_transform(hsvResult)
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(bird_eye_view, "bgr8"))

    def birdseye_transform(self, image):
      p1=[121,150] #좌상
      p2=[6,239] #좌하
      p3=[199,150] #우상
      p4=[314,239] #우하

      corner_points_arr=np.float32([p1,p3,p4,p2])
      height,width=image.shape[:2]

      image_p1=[0,0]
      image_p2=[width,0]
      image_p3=[width,height]
      image_p4=[0,height]

      image_params=np.float32([image_p1,image_p2,image_p3,image_p4])
      mat=cv2.getPerspectiveTransform(corner_points_arr,image_params)
      image_transformed=cv2.warpPerspective(image,mat,(width,height))

      return image_transformed

    def hsvExtraction(self,image):
      hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

      yellow_lower = np.array([20, 100, 100])
      yellow_upper = np.array([30, 255, 255])
      white_lower = np.array([0, 0, 200])
      white_upper = np.array([180, 255, 255])

      yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
      white_mask = cv2.inRange(hsv, white_lower, white_upper)

      combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
      masked_image = cv2.bitwise_and(image, image, mask=combined_mask)

      return masked_image

if __name__ == '__main__':
  node = LaneDetectionNode()
  rospy.spin()
