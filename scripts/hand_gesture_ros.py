#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
import copy
import argparse
import rospy
from mediapipe_ros.msg import Hand
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge

import sys

import cv2 as cv
import numpy as np
import math
import mediapipe as mp
from utils import CvFpsCalc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')
    args, unknown = parser.parse_known_args()

    return args

class Publishsers:
    def __init__(self):
        # Publisherを作成
        self.publisher = rospy.Publisher('/hand_status', Hand, queue_size=10)
        # messageの型を作成
        self.message_hand = Hand()
        # Publisherを作成
        self.publisher_image = rospy.Publisher('/cv_hand_sense', Image, queue_size=10)
        # messageの型を作成
        self.message_image = Image()
        # 引数解析 #################################################################
        args = get_args()
        max_num_hands = args.max_num_hands
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence

        self.use_brect = args.use_brect

        # # カメラ準備 ###############################################################
        # self.cap = cv.VideoCapture(cap_device)
        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        # モデルロード #############################################################
        mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # FPS計測モジュール ########################################################
        self.cvFpsCalc = CvFpsCalc(buffer_len=10)

    def make_msg(self, image):
        bridge = CvBridge()
        display_fps = self.cvFpsCalc.get()
        # カメラキャプチャ #####################################################
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self.hands.process(image)

        # 描画 ################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                # 手の平重心計算
                cx, cy = self.calc_palm_moment(debug_image, hand_landmarks)
                # 外接矩形の計算
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                # 描画
                debug_image = self.draw_landmarks(debug_image, cx, cy,
                                            hand_landmarks, handedness)
                debug_image = self.draw_bounding_rect(self.use_brect, debug_image, brect)
                distance_finger = self.distance_finger_calc(cx, cy, debug_image, hand_landmarks)
                gesture = self.gesture_detect(cx, cy, distance_finger, brect)
                if(handedness.classification[0].label[0] == 'R'):
                    cv.putText(debug_image, "Right gesture : " + str(gesture), (10, 80),
                            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(debug_image, "Right X :" + str(cx) + " Right Y : " + str(cy), (10, 120),
                            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
                    self.message_hand.right_x = cx
                    self.message_hand.right_y = cy
                    self.message_hand.right_gesture = gesture
                if(handedness.classification[0].label[0] == 'L'):
                    cv.putText(debug_image, "Left gesture : " + str(gesture), (10, 160),
                            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(debug_image, "Left X :" + str(cx) + " Left Y : " + str(cy), (10, 200),
                            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
                    self.message_hand.left_x = cx
                    self.message_hand.left_y = cy
                    self.message_hand.left_gesture = gesture

        cv.putText(debug_image, "FPS : " + str(display_fps), (10, 40),
                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        # 画面反映 #############################################################
        #cv.imshow('Hand Sensing', debug_image)
        self.message_image = bridge.cv2_to_imgmsg(debug_image, "bgr8")
        cv.waitKey(1)

    def send_msg(self):
        self.publisher.publish(self.message_hand)
        self.publisher_image.publish(self.message_image)

    def calc_palm_moment(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        palm_array = np.empty((0, 2), int)

        for index, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            if index == 0:  # 手首1
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 1:  # 手首2
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 5:  # 人差指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 9:  # 中指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 13:  # 薬指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
            if index == 17:  # 小指：付け根
                palm_array = np.append(palm_array, landmark_point, axis=0)
        M = cv.moments(palm_array)
        cx, cy = 0, 0
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        return cx, cy


    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]


    def draw_landmarks(self, image, cx, cy, landmarks, handedness):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # キーポイント
        for index, landmark in enumerate(landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue

            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append((landmark_x, landmark_y))

            if index == 0:  # 手首1
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 1:  # 手首2
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 2:  # 親指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 3:  # 親指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 4:  # 親指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 5:  # 人差指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 6:  # 人差指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 7:  # 人差指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 8:  # 人差指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 9:  # 中指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 10:  # 中指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 11:  # 中指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 12:  # 中指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 13:  # 薬指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 14:  # 薬指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 15:  # 薬指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 16:  # 薬指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
            if index == 17:  # 小指：付け根
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 18:  # 小指：第2関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 19:  # 小指：第1関節
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            if index == 20:  # 小指：指先
                cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
                cv.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

        # 接続線
        if len(landmark_point) > 0:
            # 親指
            cv.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
            cv.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)

            # 人差指
            cv.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
            cv.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
            cv.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)

            # 中指
            cv.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
            cv.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
            cv.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)

            # 薬指
            cv.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
            cv.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
            cv.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)

            # 小指
            cv.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
            cv.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
            cv.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)

            # 手の平
            cv.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
            cv.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
            cv.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
            cv.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
            cv.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
            cv.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
            cv.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

        # 重心 + 左右
        if len(landmark_point) > 0:
            # handedness.classification[0].index
            # handedness.classification[0].score

            cv.circle(image, (cx, cy), 12, (0, 255, 0), 2)
            cv.putText(image, handedness.classification[0].label[0],
                    (cx - 6, cy + 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                    2, cv.LINE_AA)  # label[0]:一文字目だけ

        return image


    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # 外接矩形
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                        (0, 255, 0), 2)

        return image

    def distance_finger_calc(self, cx, cy, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        center = [cx, cy]

        landmark_point = []

        # キーポイント
        for _ , landmark in enumerate(landmarks.landmark):
            if landmark.visibility < 0 or landmark.presence < 0:
                continue

            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append((landmark_x, landmark_y))


        distance_finger = []
        distance_finger.append(self.distance_between_points(center, landmark_point[4]))#親指の指先と重心の距離　
        distance_finger.append(self.distance_between_points(center, landmark_point[8]))#人差し指の指先と重心の距離　
        distance_finger.append(self.distance_between_points(center, landmark_point[12]))#中指の指先と重心の距離　
        distance_finger.append(self.distance_between_points(center, landmark_point[16]))#薬指の指先と重心の距離　
        distance_finger.append(self.distance_between_points(center, landmark_point[20]))#小指の指先と重心の距離　
        distance_finger.append(self.distance_between_points(center, landmark_point[3]))#親指の第1関節と重心の距離　
        distance_finger.append(self.distance_between_points(center, landmark_point[6]))#人差し指の第2関節と重心の距離　
        distance_finger.append(self.distance_between_points(center, landmark_point[10]))#中指の第2関節と重心の距離　
        distance_finger.append(self.distance_between_points(center, landmark_point[14]))#薬指の第2関節と重心の距離　
        distance_finger.append(self.distance_between_points(center, landmark_point[18]))#小指の第2関節と重心の距離　

        return distance_finger

    def gesture_detect(self, cx, cy, distance_finger, brect):
        gesture = 'None'
        is_finger_open = []
        center_x = (brect[0] + brect[2]) / 2
        center_y = (brect[1] + brect[3]) / 2
        width = brect[2] - brect[0]
        height = brect[3] - brect[1]
        for i in range(5):
            if(distance_finger[i] > distance_finger[i+5]):
                is_finger_open.append(1)
            else:
                is_finger_open.append(0)

        #print(is_finger_open)
        if(is_finger_open == [1, 1, 1, 1, 1]):
            gesture = 'release'

        elif(is_finger_open == [1, 0, 0, 0, 0] or is_finger_open == [0, 0, 0, 0, 0]):
            gesture = 'grab'

        elif(is_finger_open == [1, 1, 0, 0, 0] or is_finger_open == [0, 1, 0, 0, 0]):
            if(width > height):
                if(center_x > cx):
                    gesture = 'right'
                else:
                    gesture = 'left'
            else:
                if(center_y < cy):
                    gesture = 'up'
                else:
                    gesture = 'down'

        return gesture

    def distance_between_points(self, point1, point2):
        ans = math.sqrt(( point1[0] - point2[0] ) ** 2 + ( point1[1] - point2[1] ) ** 2)
        return ans


class Subscribe_publishers:
    def __init__(self, pub):
        self.pub = pub
        # Subscriberを作成
        rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        self.message = Hand()

    def callback(self, message):
        try:
            bridge = CvBridge()
            img = bridge.imgmsg_to_cv2(message, "bgr8")
            #cv.imshow('image', img)
            cv.waitKey(1)
        except Exception as err:
            print(err)
        # callback時の処理
        self.pub.make_msg(img)
        # publish
        self.pub.send_msg()

def main():
    # nodeの立ち上げ
    rospy.init_node('Node_name')

    # クラスの作成
    pub = Publishsers()
    sub = Subscribe_publishers(pub)

    rospy.spin()

if __name__ == '__main__':
   main()
