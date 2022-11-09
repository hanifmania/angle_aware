#!/usr/bin/env python
# -*- coding: utf-8 -*-


from bebop_aruco.ar_detect import ARDetect

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image


class ShowDetectedAR:
    def __init__(self):
        # input_img_topic = rospy.get_param("~input_img_topic", "image_raw")
        input_img_compressed_topic = rospy.get_param(
            "~input_img_compressed_topic", "image_raw/compressed"
        )
        # pose_stamped_feedback = rospy.get_param("~pose_stamped_feedback", "posestamped")
        output_topic = rospy.get_param(
            "~output_topic"
        )
        ar_params = rospy.get_param("/AR")
        self._ar_type = ar_params[
            "ar_type"
        ]  # AR markerの種類(https://docs.opencv.org/3.4/d9/d6a/group__aruco.html#gac84398a9ed9dd01306592dd616c2c975)
        self._cv_bridge = CvBridge()
        self._ar = ARDetect()
        self._ar.set_dictionary( self._ar_type)
        # self._pub_img = rospy.Publisher(
        #     output_topic, CompressedImage, queue_size=1
        # )
        self._pub_img = rospy.Publisher(
            output_topic, Image, queue_size=1
        )
        rospy.Subscriber(input_img_compressed_topic, CompressedImage, self.img_callback)
    #############################################################
    # callback
    #############################################################
    def img_callback(self, img_msg):
        """発見したARを描写

        Args:
            img_msg (sensor_msgs.msg.CompressedImage): 入力画像
        """
        cv_array = self._cv_bridge.compressed_imgmsg_to_cv2(img_msg)
        self._ar.set_img(cv_array)
        self._ar.find_marker()
        img = self._ar.get_ar_detect_img()
        compressed_img = self._cv_bridge.cv2_to_imgmsg(img)
        self._pub_img.publish(compressed_img)

if __name__ == "__main__":
    rospy.init_node("show_detected_ar", anonymous=True)
    node = ShowDetectedAR()
    rospy.spin()
