#!/usr/bin/env python
# -*- coding: utf-8 -*-

from bebop_aruco.ar_detect import ARDetect

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, CompressedImage

import numpy as np
import cv2


class Debug:
    def __init__(self):
        self._clock = rospy.get_param("~clock", 100)
        output_topic = rospy.get_param("~output_topic", "image_raw")
        # output_topic = rospy.get_param("~output_topic", "image_raw/compressed")
        output_info_topic = rospy.get_param("~output_topic", "camera_info")
        camera_matrix_csv_path = rospy.get_param("camera_matrix_csv_path")
        dist_coeffs_csv_path = rospy.get_param("dist_coeffs_csv_path")
        ar_params = rospy.get_param("/AR")
        self._ar_marker_size = ar_params["ar_marker_size"]  # ARマーカー一辺. 黒い外枠含む[m]
        self._ar_type = ar_params[
            "ar_type"
        ]  # AR markerの種類(https://docs.opencv.org/3.4/d9/d6a/group__aruco.html#gac84398a9ed9dd01306592dd616c2c975)

        camera_matrix = np.loadtxt(camera_matrix_csv_path, delimiter=",")
        distCoeffs = np.loadtxt(dist_coeffs_csv_path, delimiter=",")

        camera_info_msg = CameraInfo()
        camera_info_msg.header.frame_id = "world"
        camera_info_msg.header.stamp = rospy.Time.now()
        camera_info_msg.height = 720
        camera_info_msg.width = 1280
        camera_info_msg.D = distCoeffs
        camera_info_msg.K = camera_matrix.reshape(-1)

        self._camera_info = camera_info_msg

        self._cap = cv2.VideoCapture(0)
        self._cap.set(3, camera_info_msg.width)
        self._cap.set(4, camera_info_msg.height)

        self._ar = ARDetect()
        self._ar.set_params(
            self._ar_marker_size, self._ar_type, camera_matrix, distCoeffs
        )
        self._cv_bridge = CvBridge()
        self.dt = 1.0 / self._clock
        self._pub_img = rospy.Publisher(output_topic, Image, queue_size=1)
        self._pub_img_raw = rospy.Publisher("image_raw", Image, queue_size=1)
        self._pub_img_raw.publish(Image())
        # self._pub_img = rospy.Publisher(output_topic, CompressedImage, queue_size=1)
        self._pub_camera_info = rospy.Publisher(
            output_info_topic, CameraInfo, queue_size=1
        )

    #############################################################
    # functions
    #############################################################

    def send_img(self):
        # for _ in range(5):
        _, frame = self._cap.read()
        # t = rospy.Time.now()
        img_msg = self._cv_bridge.cv2_to_imgmsg(frame, encoding="passthrough")
        # img_msg = self._cv_bridge.cv2_to_compressed_imgmsg(frame)
        self._pub_img.publish(img_msg)
        # dt = rospy.Time.now() - t
        # rospy.loginfo(dt.to_sec())
        # self._ar.set_img(frame)
        # ids = self._ar.find_marker()
        # if ids is None:

        #     ### AR markerが見つからなければearly return
        #     return
        # im = self._ar.get_ar_detect_img()
        # img_msg = self._cv_bridge.cv2_to_imgmsg(im, encoding="passthrough")
        # self._pub_img.publish(img_msg)

    #############################################################
    # spin
    #############################################################
    def spin(self):
        rate = rospy.Rate(self._clock)
        while not rospy.is_shutdown():
            self.send_img()
            self._pub_camera_info.publish(self._camera_info)
            # rate.sleep()


if __name__ == "__main__":
    rospy.init_node("template", anonymous=True)
    node = Debug()
    node.spin()
