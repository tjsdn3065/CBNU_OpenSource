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
        self.node_name = "line_detection_node"
        rospy.init_node(self.node_name)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/camera/line_detection", Image, queue_size=1)


    def image_callback(self, img_msg):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        hsvResult=self.hsvExtraction(cv_image)
        #bird_eye_view = self.birdseye_transform(cv_image)

        canny=self.canny_image(hsvResult)
        cropped_image=self.region_of_interest(canny)
        lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
        average_lines=self.average_slope_intercept(hsvResult,lines)
        line_image= self.display_lines(hsvResult,average_lines)
        combo_image=cv2.addWeighted(hsvResult,0.8,line_image,1,1)
        #cv2.imshow('image',combo_image)
        #cv2.waitKey(0)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(combo_image, "bgr8"))

    def make_coordinates(self,image,line_parameters):
        slope,intercept=line_parameters
        y1=image.shape[0]
        y2=150
        x1=int((y1-intercept)/slope)
        x2=int((y2-intercept)/slope)
        return np.array([x1,y1,x2,y2])

    def average_slope_intercept(self,image,lines):
        left_fit=[]
        right_fit=[]
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            parameters=np.polyfit((x1,x2),(y1,y2),1)
            slope=parameters[0]
            intercept=parameters[1]
            if slope < 0:
                left_fit.append((slope,intercept))
            else:
                right_fit.append((slope,intercept))
        left_fit_average=np.average(left_fit,axis=0)
        right_fit_average=np.average(right_fit,axis=0)
        #print(left_fit_average,"L")
        #print(right_fit_average,"R")
        left_line=self.make_coordinates(image,left_fit_average)
        right_line=self.make_coordinates(image,right_fit_average)
        return np.array([left_line,right_line])



    def canny_image(self,image):
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)
        canny=cv2.Canny(blur,50,150)
        return canny

    def region_of_interest(self,image):
        height=image.shape[0]
        polygons=np.array([
            (6,height),(314,height),(160,100)
            ])
        mask=np.zeros_like(image)
        cv2.fillConvexPoly(mask,polygons,255)
        masked_image=cv2.bitwise_and(image,mask)
        return masked_image

    def display_lines(self,image,lines):
        line_image=np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2=line.reshape(4)
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),5)
        return line_image


    def birdseye_transform(self, image):
        image_height=image.shape[0]
        p1=[121,150] #좌상
        p2=[6,image_height] #좌하
        p3=[199,150] #우상
        p4=[314,image_height] #우하

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
