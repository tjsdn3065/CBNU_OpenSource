#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import matplotlib.pyplot as plt


class Line:
    def __init__(self):
        # 마지막 반복에서 선이 감지되었는가?
        self.detected = False
        # 창의 너비를 +/- 마진으로 설정
        self.window_margin = 56
        # 마지막 n번의 반복동안 적합된 선의 x 값
        self.prevx = []
        # 가장 최근 적합을 위한 다항식 계수
        self.current_fit = [np.array([False])]
        #일정 단위로 측정된 선의 곡률 반경
        self.radius_of_curvature = None
        # 시작 x 값
        self.startx = None
        # 끝 x 값
        self.endx = None
        # 감지된 선 픽셀들의 x 값
        self.allx = None
        # 감지된 선 픽셀들의 y 값
        self.ally = None

class LaneDetectionNode(Line):
    def __init__(self):
        self.node_name = "line_detection_node"
        rospy.init_node(self.node_name)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/camera/line_detection", Image, queue_size=1)

    def image_callback(self, img_msg):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8") # 이미지 메시지를 OpenCV 이미지 형식으로 변환
        frame=self.hsvExtraction(cv_image) # 노란색과 흰색 영역만을 추출
        height,width=frame.shape[:2] # 이미지의 높이와 너비
        temp=frame[0:height,:width,2] # 이미지의 전체 높이와 너비에 걸친 레드 채널만을 추출

        warp_img=self.perspective_transform(temp) #원근 변환

        left_line = Line()
        right_line = Line()

        histogram = np.sum(warp_img[int(warp_img.shape[0] / 2):, :], axis=0) # 이미지의 하단 절반에서 수평 방향으로 픽셀 값을 합산하여 히스토그램을 생성
        output = np.dstack((warp_img, warp_img, warp_img)) * 255 # 이미지를 3채널의 컬러 이미지로 변환하고 픽셀 값을 [0, 255] 범위로 조정

        # 히스토그램을 반으로 나눠 중간점을 기준으로 왼쪽, 오른쪽의 시작 포인트를 탐색
        midpoint = int(histogram.shape[0] / 2) # 히스토그램의 중간점을 계산, 이 중간점을 기준으로 히스토그램을 왼쪽과 오른쪽 두 부분으로 나눔
        start_leftX = np.argmax(histogram[:midpoint]) # 왼쪽 반쪽의 히스토그램에서 가장 높은 값을 가진 인덱스(즉, 가장 큰 빈도를 가진 위치)를 찾는다. 이 위치는 이미지의 왼쪽 차선의 시작점으로 간주
        start_rightX = np.argmax(histogram[midpoint:]) + midpoint #오른쪽 반쪽의 히스토그램에서 가장 높은 값을 가진 인덱스를 찾는다. 그리고 이 값을 중간점 값에 더하여 원래 이미지에서의 위치를 얻고 이 위치는 이미지의 오른쪽 차선의 시작점으로 간주

        # 슬라이딩 윈도우 기법을 사용하여 워핑된 이미지(warp_img) 내의 차선을 탐지하는 준비 작업
        num_windows = 10 # 슬라이딩 윈도우의 개수를 설정
        window_height = int(warp_img.shape[0] / num_windows) # 각 슬라이딩 윈도우의 높이를 계산
        nonzero = warp_img.nonzero() # 워핑된 이미지 내에서 값이 0이 아닌 모든 픽셀의 위치
        nonzeroy = np.array(nonzero[0]) # 0이 아닌 픽셀의 y좌표와 x좌표를 각각의 배열로 저장
        nonzerox = np.array(nonzero[1])
        current_leftX = start_leftX # 각 윈도우에서의 시작 x좌표를 설정
        current_rightX = start_rightX
        min_num_pixel = 50 # 슬라이딩 윈도우가 다음 위치로 이동하기 위해 필요한 최소 픽셀 수를 설정
        window_margin = left_line.window_margin # 슬라이딩 윈도우의 마진을 설정

        # 차선의 픽셀 인덱스를 수집하기 위한 두 개의 빈 리스트를 초기화
        win_left_lane = [] # 왼쪽 차선의 픽셀 인덱스를 저장하기 위한 빈 리스트
        win_right_lane = [] # 오른쪽 차선의 픽셀 인덱스를 저장하기 위한 빈 리스트

        # 슬라이딩 윈도우 방법을 사용하여 워핑된 이미지 내의 차선을 탐지
        for window in range(num_windows): # num_windows로 정의된 윈도우 개수만큼 반복
            # 윈도우의 경계를 정의
            # win_y_low와 win_y_high는 윈도우의 y축 방향의 하단 및 상단 경계
            win_y_low = warp_img.shape[0] - (window + 1) * window_height
            win_y_high = warp_img.shape[0] - window * window_height
            # win_leftx_min과 win_leftx_max는 왼쪽 차선의 x축 방향의 경계
            win_leftx_min = current_leftX - window_margin
            win_leftx_max = current_leftX + window_margin
            # win_rightx_min과 win_rightx_max는 오른쪽 차선의 x축 방향의 경계
            win_rightx_min = current_rightX - window_margin
            win_rightx_max = current_rightX + window_margin

            # 윈도우의 경계를 시각화
            # cv2.rectangle 함수를 사용하여 output 이미지에 윈도우의 경계를 그림
            cv2.rectangle(output, (win_leftx_min, win_y_low), (win_leftx_max, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(output, (win_rightx_min, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)

            # 각 윈도우 내의 0이 아닌 픽셀을 찾는다
            # left_window_inds와 right_window_inds는 각각 왼쪽 및 오른쪽 윈도우 내의 0이 아닌 픽셀의 인덱스를 반환
            left_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_leftx_min) & (
                nonzerox <= win_leftx_max)).nonzero()[0]
            right_window_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high) & (nonzerox >= win_rightx_min) & (
                nonzerox <= win_rightx_max)).nonzero()[0]
            # 찾은 픽셀의 인덱스를 win_left_lane 및 win_right_lane에 추가
            win_left_lane.append(left_window_inds)
            win_right_lane.append(right_window_inds)

            # 만약 윈도우 내에서 min_num_pixel보다 많은 픽셀이 발견되면, 다음 윈도우의 중심을 해당 픽셀들의 평균 위치로 이동
            if len(left_window_inds) > min_num_pixel:
                current_leftX = int(np.mean(nonzerox[left_window_inds]))
            if len(right_window_inds) > min_num_pixel:
                current_rightX = int(np.mean(nonzerox[right_window_inds]))

        # 각각의 슬라이딩 윈도우 내에서 발견된 차선 픽셀의 인덱스들을 수집하고, 그 정보를 기반으로 실제 차선의 픽셀 위치들을 추출하는 과정
        # 여러 윈도우 내에서 감지된 픽셀들의 인덱스들이 리스트 형태로 저장
        # np.concatenate를 사용하여 이러한 여러 개의 리스트들을 하나의 배열로 결합
        win_left_lane = np.concatenate(win_left_lane)
        win_right_lane = np.concatenate(win_right_lane)

        # 왼쪽 및 오른쪽 차선의 x, y 좌표 픽셀 위치를 추출
        # nonzerox와 nonzeroy는 워핑된 이미지에서 0이 아닌 픽셀의 x, y 좌표
        # win_left_lane 및 win_right_lane를 인덱스로 사용하여 해당 차선의 픽셀 위치를 가져옴
        leftx, lefty = nonzerox[win_left_lane], nonzeroy[win_left_lane]
        rightx, righty = nonzerox[win_right_lane], nonzeroy[win_right_lane]

        # output 이미지에서 감지된 왼쪽 차선 픽셀들을 빨간색으로([255, 0, 0]) 표시하고, 오른쪽 차선 픽셀들을 파란색으로([0, 0, 255]) 표시
        output[lefty, leftx] = [255, 0, 0]
        output[righty, rightx] = [0, 0, 255]

        # 왼쪽 및 오른쪽 차선의 픽셀에 2차 다항식을 적합하고 해당 정보를 Line 객체에 저장하는 과정
        # 1. 다항식 적합
        # np.polyfit 함수를 사용하여 왼쪽 및 오른쪽 차선 픽셀에 2차 다항식을 적합
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # 2. 저장 및 갱신
        # 각 차선의 현재 적합 결과는 current_fit 속성에 저장
        left_line.current_fit = left_fit
        right_line.current_fit = right_fit

        # warp_img 이미지의 높이만큼의 y 좌표를 생성
        ploty = np.linspace(0, warp_img.shape[0] - 1, warp_img.shape[0])

        # 3. 다항식 값 계산
        # 워핑된 이미지의 높이에 해당하는 y 값을 사용하여 x 좌표를 생성
        left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # left_line.prevx와 right_line.prevx에 각각 left_plotx와 right_plotx 값을 추가
        left_line.prevx.append(left_plotx)
        right_line.prevx.append(right_plotx)

        # 4. 평활화
        # 연속된 프레임에서 감지된 차선을 더 부드럽게 만들기 위해 smoothing 함수를 사용하여 최근 10개의 차선 x 좌표를 평균화
        # 이 평균화된 차선은 다시 다항식으로 적합되고, 그 결과는 current_fit 속성에 저장
        if len(left_line.prevx) > 10:
            left_avg_line = smoothing(left_line.prevx, 10)
            left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
            left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
            left_line.current_fit = left_avg_fit
            left_line.allx, left_line.ally = left_fit_plotx, ploty
        else:
            left_line.current_fit = left_fit
            left_line.allx, left_line.ally = left_plotx, ploty

        if len(right_line.prevx) > 10:
            right_avg_line = smoothing(right_line.prevx, 10)
            right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
            right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
            right_line.current_fit = right_avg_fit
            right_line.allx, right_line.ally = right_fit_plotx, ploty
        else:
            right_line.current_fit = right_fit
            right_line.allx, right_line.ally = right_plotx, ploty

        # 5. 차선 시작 및 종료 위치 저장
        left_line.startx, right_line.startx = left_line.allx[len(left_line.allx)-1],right_line.allx[len(right_line.allx)-1]
        left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

        # 6. 차선 감지 플래그 업데이트
        # 차선이 성공적으로 감지되었음을 나타내는 detected 속성을 True로 설정
        left_line.detected, right_line.detected = True, True

        """ 곡률 반경 측정  """
        # 곡률 계산
        # 화면 상의 차선 픽셀 위치를 실제 세계 좌표(미터 단위)로 변환하는 작업
        # 1. 차선 y 및 x 좌표 추출
        ploty = left_line.ally
        leftx, rightx = left_line.allx, right_line.allx

        # 2. 차선 좌표 반전
        # 차선 x 좌표의 순서를 상하 반전(y 좌표가 상단에서 하단으로 나타나는 순서와 일치하게 만들기 위함)
        leftx = leftx[::-1]
        rightx = rightx[::-1]

        # 3. 픽셀에서 미터로의 변환 설정
        width_lanes = abs(right_line.startx - left_line.startx)
        ym_per_pix = 30 / 720 #  y축 방향의 픽셀에서 미터로의 변환 비율(720 픽셀이 30미터에 해당한다고 가정)
        xm_per_pix = 3.7*(720/1280) / width_lanes # x축 방향의 픽셀에서 미터로의 변환 비율((720/1280) 비율은 화면의 종횡비를 조정하기 위해 사용)

        # 차선의 곡률 반경을 계산하는 과정
        # 1. 곡률 반경을 계산할 y값 정의
        y_eval = np.max(ploty) # 이미지의 바닥 부분에 해당하는 y값에서 곡률 반경을 계산하기 위해 최대 y값을 선택

        # 2. 세계 좌표계에서의 새로운 다항식 적합
        # left_fit_cr와 right_fit_cr: 차선의 y와 x 좌표를 미터 단위로 변환한 후 2차 다항식으로 적합
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # 3. 곡률 반경 계산
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        # 4. 곡률 반경 결과 저장
        left_line.radius_of_curvature = left_curverad
        right_line.radius_of_curvature = right_curverad

        """
       차선 탐지에 실패한 경우 → 이전에 탐지된 window가 있다면 이를 기반으로 차선을 그림
        """
        # 이전 프레임의 차선 정보를 사용하여 현재 프레임에서 차선을 더 효과적으로 탐지
        # 이미지에서 0이 아닌 모든 픽셀의 x 및 y 위치 확인
        nonzero = warp_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # 창의 마진 설정
        window_margin = left_line.window_margin

        # 이전 프레임에서 차선의 다항식 적합 가져오기
        left_line_fit = left_line.current_fit
        right_line_fit = right_line.current_fit

        # 차선 주변의 검색 영역 설정
        leftx_min = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
        leftx_max = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
        rightx_min = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
        rightx_max = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

        # 검색 영역 내에서 0이 아닌 픽셀 확인
        left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]
        right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]

        # 차선 픽셀 위치 추출
        leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
        rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

        # 시각화
        output[lefty, leftx] = [255, 0, 0]
        output[righty, rightx] = [0, 0, 255]

        """ 차선관 주행 공간 시각화 """
        # 이전에 검출된 차선과 현재 운전 공간을 시각화
        # 1. 초기화
        window_img = np.zeros_like(output) # 출력 이미지와 동일한 크기의 검은색 이미지를 생성
        # 각각 차선 주변의 검색 영역 폭과 왼쪽/오른쪽 차선의 x 좌표, 그리고 y 좌표를 가져옴
        window_margin = left_line.window_margin
        left_plotx, right_plotx = left_line.allx, right_line.allx
        ploty = left_line.ally

        # 2. 차선의 폴리곤(polygon) 생성
        # 검색 창 영역을 나타내기 위한 다각형 생성
        # 그리고 x와 y 점들을 cv2.fillPoly()에서 사용 가능한 형식으로 다시 변환
        left_pts_l = np.array([np.transpose(np.vstack([left_plotx - window_margin/5, ploty]))]) # 차선의 왼쪽 경계
        left_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_plotx + window_margin/5, ploty])))]) # 차선의 오른쪽 경계
        left_pts = np.hstack((left_pts_l, left_pts_r)) # 좌/우 경계를 결합하여 각 차선에 대한 폴리곤을 형성

        right_pts_l = np.array([np.transpose(np.vstack([right_plotx - window_margin/5, ploty]))]) # 차선의 왼쪽 경계
        right_pts_r = np.array([np.flipud(np.transpose(np.vstack([right_plotx + window_margin/5, ploty])))]) # 차선의 오른쪽 경계
        right_pts = np.hstack((right_pts_l, right_pts_r)) # 좌/우 경계를 결합하여 각 차선에 대한 폴리곤을 형성

        # 3. 차선 그리기
        # 보라색을 사용하여 차선을 그림
        cv2.fillPoly(window_img, np.int_([left_pts]), (140, 0, 170))
        cv2.fillPoly(window_img, np.int_([right_pts]), (140, 0, 170))

        # 4. 운전 공간의 폴리곤 생성 및 그리기
        # pts_left 및 pts_right은 현재 운전 공간을 나타내는 폴리곤의 좌/우 경계를 형성
        pts_left = np.array([np.transpose(np.vstack([left_plotx+window_margin/5, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx-window_margin/5, ploty])))])
        pts = np.hstack((pts_left, pts_right)) # 운전 공간의 폴리곤을 초록색으로 그림
        cv2.fillPoly(window_img, np.int_([pts]), (0, 160, 0))

        # 5. 결과 합성
        lane_result = cv2.addWeighted(output, 1, window_img, 0.3, 0)

        Minv=self.get_Minv(temp)

        ''' 최종 이미지 합성 '''

        # 차선을 원래의 시점으로 되돌리고, 원본 프레임과 합성
        # 1. 투영 변환의 역 연산
        color_result = cv2.warpPerspective(window_img, Minv, (width, height))

        # 2. 차선을 원본 프레임에 추가
        lane_color = np.zeros_like(frame) # 원래 프레임과 동일한 크기의 검은색 이미지 lane_color를 생성
        lane_color += color_result

        # 3. 원본 프레임과 합성
        res = cv2.addWeighted(cv_image, 1, lane_color, 1, 0)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(res, "bgr8"))


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

    def get_Minv(self,img):
        height, width = img.shape[:2]
        s_LTop2, s_RTop2 = [80, 160], [240, 160]
        s_LBot2, s_RBot2 = [40, 230], [280, 230]

        src = np.float32([s_LTop2, s_RTop2, s_RBot2, s_LBot2])
        dst = np.float32([(250, 0), (510, 0), (510, 720), (250, 720)])

        Minv = cv2.getPerspectiveTransform(dst, src)
        return Minv


if __name__ == '__main__':
    node = LaneDetectionNode()
    rospy.spin()
