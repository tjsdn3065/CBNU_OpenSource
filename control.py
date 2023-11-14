#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

class PIDController:
    def __init__(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class Line:
    def __init__(self):
        self.window_margin = 56

class ControlNode(Line):
    def __init__(self):
        self.node_name = "control_node"
        rospy.init_node(self.node_name)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.pid_controller = PIDController(0.045, 0.0007, 0.15) # 조정 필요

    def image_callback(self, img_msg):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8") # 이미지 메시지를 OpenCV 이미지 형식으로 변환
        frame=self.hsvExtraction(cv_image) # 노란색과 흰색 영역만을 추출
        height,width=frame.shape[:2] # 이미지의 높이와 너비
        temp=frame[0:height,:width,2] # 이미지의 전체 높이와 너비에 걸친 레드 채널만을 추출

        warp_img=self.perspective_transform(temp) #원근 변환

        leftx,rightx,output=self.findpoint(warp_img)

        # 중앙 차선 위치 계산
        if leftx.size and rightx.size:  # 두 차선이 모두 감지된 경우
            lane_center = (leftx[0] + rightx[0]) / 2
        elif leftx.size:  # 오직 왼쪽 차선만 감지된 경우
            lane_center = leftx[0] + 320
        elif rightx.size:  # 오직 오른쪽 차선만 감지된 경우
            lane_center = rightx[0] - 320
        else:
            lane_center = None

        if lane_center:
            img_center = output.shape[1] / 2
            error = img_center - lane_center

            # PID 제어
            pid_output = self.pid_controller.compute(error)

            # cmd_vel 메시지 생성 및 발행
            twist = Twist()
            twist.linear.x = 0.1  # 일정한 속도로 전진
            twist.angular.z = pid_output  # PID 출력을 기반으로 회전
            self.cmd_vel_pub.publish(twist)

    def findpoint(self,warp_img):
        left_line = Line()
        right_line = Line()

        histogram = np.sum(warp_img[int(warp_img.shape[0] / 2):, :], axis=0)
        output = np.dstack((warp_img, warp_img, warp_img)) * 255

        midpoint = int(histogram.shape[0] / 2)
        start_leftX = np.argmax(histogram[:midpoint])
        start_rightX = np.argmax(histogram[midpoint:]) + midpoint

        num_windows = 10
        window_height = int(warp_img.shape[0] / num_windows)
        nonzero = warp_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        current_leftX = start_leftX
        current_rightX = start_rightX
        min_num_pixel = 50
        window_margin = left_line.window_margin

        win_left_lane = []
        win_right_lane = []

        for window in range(num_windows):
            win_y_low = warp_img.shape[0] - (window + 1) * window_height
            win_y_high = warp_img.shape[0] - window * window_height
            win_leftx_min = current_leftX - window_margin
            win_leftx_max = current_leftX + window_margin
            win_rightx_min = current_rightX - window_margin
            win_rightx_max = current_rightX + window_margin

            cv2.rectangle(output, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(output, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

            left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
                nonzerox <= win_leftx_max)).nonzero()[0]
            right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
                nonzerox <= win_rightx_max)).nonzero()[0]
            win_left_lane.append(left_window_inds)
            win_right_lane.append(right_window_inds)

            if len(left_window_inds) > min_num_pixel:
                current_leftX = int(np.mean(nonzerox[left_window_inds]))
            if len(right_window_inds) > min_num_pixel:
                current_rightX = int(np.mean(nonzerox[right_window_inds]))

        win_left_lane = np.concatenate(win_left_lane)
        win_right_lane = np.concatenate(win_right_lane)

        leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
        rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

        output[lefty, leftx] = [255, 0, 0]
        output[righty, rightx] = [0, 0, 255]

        return leftx, rightx, output

    def hsvExtraction(self,image): #주어진 이미지에서 HSV 색상 공간을 이용하여 노란색과 흰색 영역만 추출하는 함수
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        yellow_lower = np.array([10, 70, 95])
        yellow_upper = np.array([127, 255, 255])
        white_lower = np.array([0, 0, 105])
        white_upper = np.array([179, 70, 255])

        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
        masked_image = cv2.bitwise_and(image, image, mask=combined_mask)

        return masked_image

    def perspective_transform(self,img): #이미지에 대한 원근 변환(Perspective Transformation)을 수행하는 함수
        height, width = img.shape[:2]
        s_LTop2, s_RTop2 = [80, 160], [240, 160]
        s_LBot2, s_RBot2 = [40, 230], [280, 230]

        src = np.float32([s_LTop2, s_RTop2, s_RBot2, s_LBot2])
        dst = np.float32([(250, 0), (510, 0), (510, 720), (250, 720)])

        M = cv2.getPerspectiveTransform(src, dst)
        warp_img = cv2.warpPerspective(img, M, (720, 720), flags=cv2.INTER_LINEAR)
        return warp_img

if __name__ == '__main__':
    node = ControlNode()
    rospy.spin()
