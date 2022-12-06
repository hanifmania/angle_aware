#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base import ros_utility


import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo, CompressedImage

import numpy as np
import cv2
import torch


class Template:
    def __init__(self):
        ################## param input
        ###### private pram
        input_img_compressed_topic = rospy.get_param(
            "~input_img_compressed_topic", "image_raw/compressed"
        )
        input_pose_topic = rospy.get_param("~input_pose_topic", "camera_posestamped")
        output_pose_topic = rospy.get_param(
            "~output_pose_topic", "object_detector/posestamped"
        )
        output_img_topic = rospy.get_param("~output_img_topic")
        pytorch_hub_path = rospy.get_param("pytorch_hub_path")
        model_path = rospy.get_param("model_path")
        camera_matrix_param = rospy.get_param("~camera_matrix")

        K = camera_matrix_param["data"]
        ### default ok
        self._frame_id = rospy.get_param("~frame_id", "world")
        ###### group yaml

        ##### central yaml
        self._ref_z = rospy.get_param("/agents/ref_z")
        object_detector_params = rospy.get_param("/object_detector")
        confidence_min = object_detector_params["confidence_min"]
        target_class = object_detector_params["target_class"]
        self._input_img_size = object_detector_params["input_img_size"]
        ##################

        self._model = torch.hub.load(
            pytorch_hub_path, "custom", path=model_path, source="local"
        )

        self._model.conf = confidence_min  # --- 検出の下限値（<1）。設定しなければすべて検出
        self._model = self.set_target_class(self._model, target_class)

        self._cv_bridge = CvBridge()
        self._camera_matrix = np.array(K).reshape(3, -1)
        self._camera_matrix_inv = self._camera_matrix**-1

        self._pub_posestamped = rospy.Publisher(
            output_pose_topic, PoseStamped, queue_size=1
        )
        self._pub_img = rospy.Publisher(output_img_topic, Image, queue_size=1)
        rospy.Subscriber(
            input_img_compressed_topic,
            CompressedImage,
            self.img_callback,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.Subscriber(
            input_pose_topic,
            PoseStamped,
            self.posestamped_callback,
            queue_size=1,
        )

    #############################################################
    # callback
    #############################################################
    def posestamped_callback(self, msg):
        self._camera_posestamped = msg

    def img_callback(self, img_msg):
        cv_array = self._cv_bridge.compressed_imgmsg_to_cv2(img_msg)
        self.object_detect(cv_array)

    #############################################################
    # main function
    #############################################################
    def object_detect(self, cv_img):
        """image_rawからカメラ位置を推定する

        Args:
            img_msg (sensor_msgs.msg.Image): _description_
        """
        raw_img_shape = cv_img.shape
        resized_img = cv2.resize(cv_img, (self._input_img_size, self._input_img_size))
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        results = self._model([rgb_img], size=self._input_img_size)

        result_img = self.draw(resized_img, results, self._model.names)
        output_img = cv2.resize(result_img, (raw_img_shape[1], raw_img_shape[0]))
        self.publish_img(output_img)

        self.publish_posestamped()

    #############################################################
    # publish
    #############################################################
    def publish_posestamped(self):
        posestamped = PoseStamped()
        posestamped.header.frame_id = self._frame_id
        posestamped.header.stamp = rospy.Time.now()
        self._pub_posestamped.publish(posestamped)

    def publish_img(self, cv2_img):
        compressed_img = self._cv_bridge.cv2_to_imgmsg(cv2_img)
        self._pub_img.publish(compressed_img)

    #############################################################
    # functions
    #############################################################
    def inverse_dic(dictionary):
        return {v: k for k, v in dictionary.items()}

    def set_target_class(self, model, class_name):
        print(model.names)  # --- （参考）クラスの一覧をコンソールに表示
        inversed_name = self.inverse_dic(model.names)
        model.classes = [inversed_name[class_name]]
        return model

    def draw(self, img, results, class_dict):
        """検出結果の画像を生成

        Args:
            img (cv2): modelに入れた画像のcv2 version
            results (_type_): model output
            class_dict (dict): model.name

        Returns:
            cv2: 検出結果を描写した画像
        """
        for *box, conf, cls in results.xyxy[0]:  # xyxy, confidence, class
            # --- クラス名と信頼度を文字列変数に代入
            s = class_dict[int(cls)] + ":" + "{:.3f}".format(float(conf))

            # --- ヒットしたかどうかで枠色（cc）と文字色（cc2）の指定
            # if int(box[0]) > pos_x:
            #     cc = (255, 255, 0)
            #     cc2 = (128, 0, 0)
            # else:
            cc = (0, 255, 255)
            cc2 = (0, 128, 128)
            # --- 枠描画
            cv2.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color=cc,
                thickness=2,
            )
            # --- 文字枠と文字列描画
            cv2.rectangle(
                img,
                (int(box[0]), int(box[1]) - 20),
                (int(box[0]) + len(s) * 10, int(box[1])),
                cc,
                -1,
            )
            cv2.putText(
                img,
                s,
                (int(box[0]), int(box[1]) - 5),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                cc2,
                1,
                cv2.LINE_AA,
            )
        return img

    def calc_pose(self, results, camera_from_world_posestamped):
        for *box, conf, cls in results.xyxy[0]:  # xyxy, confidence, class
            center_x = (box[0] + box[2]) * 0.5
            center_y = (box[1] + box[3]) * 0.5

            screen_point = np.array([center_x, center_y, 1]).T
            object_point_from_camera = self._ref_z * self._camera_matrix_inv.dot(
                screen_point
            )
            object_point_from_camera_q = np.vstack([object_point_from_camera, [1]])
            camera_from_world = ros_utility.pose_to_g(
                camera_from_world_posestamped.pose
            )
            camera2world = camera_from_world.dot(object_point_from_camera_q)
            break  ### 一番信頼度が高いもののみをpublish

    #############################################################
    # spin
    #############################################################

    def spin(self):
        rate = rospy.Rate(self._clock)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("template", anonymous=True)
    node = Template()
    rospy.spin()
