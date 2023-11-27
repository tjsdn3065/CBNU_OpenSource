#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.prev_error = 0
        self.integral = 0

    def left_compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

    def right_compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class CameraNode:
    def __init__(self):
        self.node_name = "camera_node"
        rospy.init_node(self.node_name)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.pid_controller = PIDController(0.045, 0.0007, 0.15)

        self.obstacle_sub = rospy.Subscriber("obstacle", Bool, self.obstacle_callback)
        self.obstacle_detected = False
        self.lane_detection_flag = 2
        self.rotating = False  # 라이다 제어 실행 중인지 여부를 나타내는 플래그
        self.obstacle_count=0 # 장애물이 연속적으로 감지된 횟수
        self.variable = 2

    def image_callback(self, img_msg):
        rospy.loginfo(self.lane_detection_flag)
        if self.obstacle_detected==False:
            #rospy.loginfo("No obstacle_detected")
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            frame = self.hsvExtraction(cv_image)
            height, width = frame.shape[:2]
            temp = frame[0:height, :width, 2]

            warp_img = self.perspective_transform(temp)
            lane_center = self.findpoint(warp_img)
            if self.variable==111 or self.variable==222:
                self.go()
            else:
                self.follow_line(lane_center)
        else:
            #rospy.loginfo("Obstacle_detected")
            pass
    def go(self):
        # cmd_vel 메시지 생성 및 발행
        twist = Twist()
        twist.linear.x = 0.07  # 일정한 속도로 전진
        twist.angular.z = 0.0  # PID 출력을 기반으로 회전
        self.cmd_vel_pub.publish(twist)

    def obstacle_callback(self, msg):
        self.obstacle_detected = msg.data

        if self.obstacle_detected:
            self.obstacle_count += 1
            if self.obstacle_count >= 3 and not self.rotating:  # 라이다 제어 시작
                self.rotating = True
                self.rotate_and_drive()
        else:
            if self.rotating:  # 라이다 제어 종료 후 차선 플래그 업데이트
                self.rotating = False
                self.update_lane_detection_flag()
            self.obstacle_count = 0

    def rotate_and_drive(self):
        # 로봇을 회전하며 직진
        twist = Twist()
        twist.linear.x = 0.07  # 전진 속도
        if self.lane_detection_flag==1:
            twist.angular.z = -1.4
        elif self.lane_detection_flag==2:
            twist.angular.z = 1.4
        self.cmd_vel_pub.publish(twist)

    def update_lane_detection_flag(self):
        # 차선 감지 플래그 변경 로직
        if self.lane_detection_flag==1:
            self.lane_detection_flag=2
        elif self.lane_detection_flag==2:
            self.lane_detection_flag=1

    def follow_line(self,lane_center):
        img_center = 360
        error = img_center - lane_center

        # PID 제어
        if self.lane_detection_flag==1:
            pid_output = self.pid_controller.left_compute(error)
        else:
            pid_output = self.pid_controller.right_compute(error)

        # cmd_vel 메시지 생성 및 발행
        twist = Twist()
        twist.linear.x = 0.07  # 일정한 속도로 전진
        twist.angular.z = pid_output  # PID 출력을 기반으로 회전
        #rospy.loginfo(twist.angular.z)
        self.cmd_vel_pub.publish(twist)


    def findpoint(self,warp_img):
        histogram = np.sum(warp_img[int(warp_img.shape[0] / 2):, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)

        start_leftX = np.argmax(histogram[:midpoint]) if self.lane_detection_flag == 1 else 0
        start_rightX = np.argmax(histogram[midpoint:]) + midpoint if self.lane_detection_flag == 2 else 0

        num_windows = 10
        window_height = int(warp_img.shape[0] / num_windows)
        nonzero = warp_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        current_leftX = start_leftX
        current_rightX = start_rightX
        min_num_pixel = 50
        window_margin = 56

        win_left_lane = []
        win_right_lane = []

        for window in range(num_windows):
            win_y_low = warp_img.shape[0] - (window + 1) * window_height
            win_y_high = warp_img.shape[0] - window * window_height
            win_leftx_min = current_leftX - window_margin
            win_leftx_max = current_leftX + window_margin
            win_rightx_min = current_rightX - window_margin
            win_rightx_max = current_rightX + window_margin

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

        # 차선 간격
        line_diff=265

        # 차선 감지 여부에 따른 lane_center 계산
        if self.lane_detection_flag == 1 and len(win_left_lane) == 0:
            # 왼쪽 차선 감지 실패 시 직진
            lane_center = 360
            self.variable = 111
        elif self.lane_detection_flag == 2 and len(win_right_lane) == 0:
            # 오른쪽 차선 감지 실패 시 직진
            lane_center = 360
            self.variable = 222
        elif self.lane_detection_flag == 1:
            # 왼쪽 차선 감지
            leftx = nonzerox[win_left_lane]
            lane_center = leftx[0] + line_diff / 2 -30
            self.variable = 1
        elif self.lane_detection_flag == 2:
            # 오른쪽 차선 감지
            rightx = nonzerox[win_right_lane]
            lane_center = rightx[0] - line_diff/2 +10
            self.variable = 2
        else:
            # 차선 감지 없음
            lane_center = 360
            self.variable = 0

        # 플래그 값 출력 또는 로깅
        #rospy.loginfo(f"Lane Detection Flag: {self.lane_detection_flag}")
        #os.system('clear')

        return lane_center

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
        s_LTop2, s_RTop2 = [width-270, 200], [270, 200]
        s_LBot2, s_RBot2 = [width-280, 230], [280, 230]

        src = np.float32([s_LTop2, s_RTop2, s_RBot2, s_LBot2])
        dst = np.float32([(250, 0), (510, 0), (510, 720), (250, 720)])

        M = cv2.getPerspectiveTransform(src, dst)
        warp_img = cv2.warpPerspective(img, M, (720, 720), flags=cv2.INTER_LINEAR)
        return warp_img

if __name__ == '__main__':
    node = CameraNode()
    rospy.spin()
