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
import matplotlib.pyplot as plt

class YoloResult

class ObjectDetectorYOLOv5:
    def __init__(self):
        ################## param input
        ###### private pram
        pytorch_hub_path = rospy.get_param("~pytorch_hub_path")
        model_path = rospy.get_param("~model_path")
        camera_matrix_param = rospy.get_param("~camera_matrix")

        ### default ok
        input_img_compressed_topic = rospy.get_param(
            "~input_img_compressed_topic", "image_raw/compressed"
        )
        input_pose_topic = rospy.get_param("~input_pose_topic", "camera_posestamped")
        output_pose_topic = rospy.get_param(
            "~output_pose_topic", "object_detector/posestamped"
        )
        output_img_topic = rospy.get_param(
            "~output_img_topic", "object_detector/image_raw"
        )
        self._frame_id = rospy.get_param("~frame_id", "world")
        ###### group yaml

        ##### central yaml
        self._ref_z = rospy.get_param("/agents/ref_z")
        object_detector_params = rospy.get_param("/object_detector")
        confidence_min = object_detector_params["confidence_min"]
        target_class = object_detector_params["target_class"]
        self._model_input_size = object_detector_params["model_input_size"]
        ##################
        K = camera_matrix_param["data"]

        self._model = torch.hub.load(
            pytorch_hub_path, "custom", path=model_path, source="local"
        )

        self._model.conf = confidence_min  # --- 検出の下限値（<1）。設定しなければすべて検出
        self._model = self.set_target_class(self._model, target_class)

        self._cv_bridge = CvBridge()
        self._camera_matrix = np.array(K).reshape(3, -1)
        self._camera_matrix_inv = np.linalg.inv(self._camera_matrix)
        self._pub_posestamped = rospy.Publisher(
            output_pose_topic, PoseStamped, queue_size=1
        )
        self._pub_img = rospy.Publisher(output_img_topic, Image, queue_size=1)

        rospy.wait_for_message(input_pose_topic, PoseStamped)

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
        """imageから物体位置を推定し、publish

        Args:
            cv_img (cv2): _description_
        """
        result_df = self.detect(cv_img, self._model, self._model_input_size)
        if result_df is None:
            # 未検出なら何の描写もせずに画像のみ送信
            self.publish_img(cv_img)
            return
        output_img = self.draw(cv_img, result_df)
        self.publish_img(output_img)
        positions = self.calc_position(result_df, self._camera_posestamped)
        for position in positions:
            self.publish_posestamped(position)

        # raw_img_shape = cv_img.shape
        # resized_img = cv2.resize(
        #     cv_img, (self._model_input_size, self._model_input_size)
        # )
        # rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        # results = self._model([rgb_img], size=self._model_input_size)

        # result_img = self.draw(resized_img, results, self._model.names)
        # output_img = cv2.resize(result_img, (raw_img_shape[1], raw_img_shape[0]))
        # self.publish_img(output_img)

        # position = self.calc_position(results, self._camera_posestamped)
        # if position is not None:
        #     self.publish_posestamped(position)

    #############################################################
    # publish
    #############################################################
    def publish_posestamped(self, position):
        posestamped = PoseStamped()
        posestamped.header.frame_id = self._frame_id
        posestamped.header.stamp = rospy.Time.now()
        posestamped.pose.position.x = position[0]
        posestamped.pose.position.y = position[1]
        posestamped.pose.position.z = position[2]
        posestamped.pose.orientation.w = 1
        self._pub_posestamped.publish(posestamped)

    def publish_img(self, cv2_img):
        compressed_img = self._cv_bridge.cv2_to_imgmsg(cv2_img)
        self._pub_img.publish(compressed_img)

    #############################################################
    # functions
    #############################################################
    def detect(self, cv_img, model, model_input_size):
        """AIで入力画像から物体を検出

        Args:
            cv_img (cv2): 入力生画像
            model(Model): AI
            model_input_size (int): 画像を縦横この数値のpixelにしてAIに入力

        Returns:
            pandas.DataFrame:  [x_min y_min x_max y_max confidence  class    name]
        """
        raw_img_shape = cv_img.shape
        resized_img = cv2.resize(cv_img, (model_input_size, model_input_size))
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        results = model([rgb_img], size=model_input_size)
        # if len(results.xyxy[0]) == 0:
        #     ### 解が無ければNone
        #     return None
        
        result_df = results.pandas().xyxy[0]
        
        #### rescale box
        height = raw_img_shape[0]
        width = raw_img_shape[1]

        model_size_to_raw_x = width / model_input_size
        model_size_to_raw_y = height / model_input_size
        result_df["x_min"] *=  model_size_to_raw_x
        result_df["x_max"] *=  model_size_to_raw_x
        result_df["y_min"] *=  model_size_to_raw_y
        result_df["y_max"] *=  model_size_to_raw_y
        return result_df
        # raw_img_box = np.zeros(4)
        # raw_img_box[0] = box[0] * model_size_to_raw_x
        # raw_img_box[1] = box[1] * model_size_to_raw_y
        # raw_img_box[2] = box[2] * model_size_to_raw_x
        # raw_img_box[3] = box[3] * model_size_to_raw_y
        # return raw_img_box, confidence, cls

    def draw(self, img, box, df):
        """検出結果の画像を生成

        Args:
            img (cv2): modelに入れた画像のcv2 version
            df (pandas.DataFrame):  [x_min y_min x_max y_max confidence  class    name]

        Returns:
            cv2: 検出結果を描写した画像
        """
        cc = (0, 255, 255)
        cc2 = (0, 128, 128)
        for index, data in df.iterrows():
            name = data["name"]
            confidence = data["confidence"]
            s = name + ":" + "{:.3f}".format(float(confidence))
            x_min = int(data["x_min"])
            x_max = int(data["x_max"])
            y_min = int(data["y_min"])
            y_max = int(data["y_max"])
            
            # color_rgba = plt.get_cmap("jet")(confidence)
            # color_bgr = [color_rgba[2], color_rgba[1], color_rgba[0]]
            cv2.rectangle(
                img,
                (x_min,y_min),
                (x_max, y_max),
                color=cc,
                thickness=2,
            )

            # --- 文字枠と文字列描画
            cv2.rectangle(
                img,
                (x_min, y_min - 20),
                (x_min + len(s) * 10, y_min),
                cc,
                -1,
            )
            cv2.putText(
                img,
                s,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                cc2,
                1,
                cv2.LINE_AA,
            )
        return img

    def calc_position(self, df, camera_posestamped):
        """検出結果から位置を推定

        Args:
            df (pandas.DataFrame):  [x_min y_min x_max y_max confidence class name]
            camera_posestamped (posestamped): bebop camera posestamped

        Returns:
            list: [[x, y, z], ...]
        """
        positions = []
        for index, data in df.iterrows():
            x_min = data["x_min"]
            x_max = data["x_max"]
            y_min = data["y_min"]
            y_max = data["y_max"]
            center_x = (x_min + x_max) * 0.5
            center_y = (y_min + y_max) * 0.5

            screen_point = np.array([center_x, center_y, 1]).reshape(-1, 1)
            # rospy.loginfo("screen_point")
            # rospy.loginfo(screen_point)
            object_point_from_screen = self._ref_z * self._camera_matrix_inv.dot(
                screen_point
            )
            object_point_from_screen_q = np.vstack([object_point_from_screen, 1])
            # rospy.loginfo("object_point_from_camera_q")
            # rospy.loginfo(object_point_from_screen_q)
            bebopcamera2world = ros_utility.pose_to_g(camera_posestamped.pose)

            bebop_camera_R = bebopcamera2world[:3, :3]
            screen_R = self.change_axis(bebop_camera_R)

            screen2world = bebopcamera2world
            screen2world[:3, :3] = screen_R

            object_from_world = screen2world.dot(object_point_from_screen_q)
            position = object_from_world.reshape(-1)
            positions.append(position)
        return positions

    def inverse_dic(self, dictionary):
        """dictのkeyとvalueを入れ替える

        Args:
            dictionary (dict): _description_

        Returns:
            dict: _description_
        """
        return {v: k for k, v in dictionary.items()}

    def set_target_class(self, model, class_name):
        """モデルの検出対象物体を設定

        Args:
            model (Model): _description_
            class_name (str): _description_

        Returns:
            Model: model
        """
        print(model.names)
        inversed_name = self.inverse_dic(model.names)
        model.classes = [inversed_name[class_name]]
        return model

    def change_axis(self, R):
        """bebop2droneの座標系をcvのカメラ座標に変換

        Args:
            R (ndarray): bebop2が想定するカメラの回転行列

        Returns:
            ndarray: cvが想定するカメラの回転行列
        """
        # y軸周りに回転
        y = np.pi / 2
        y_rotate = np.array(
            [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
        )
        # z軸周りに回転
        z = -np.pi / 2
        z_rotate = np.array(
            [[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]]
        )
        R2 = R.dot(y_rotate)
        bebop_R = R2.dot(z_rotate)
        return bebop_R


if __name__ == "__main__":
    rospy.init_node("ObjectDetectorYOLOv5", anonymous=True)
    node = ObjectDetectorYOLOv5()
    rospy.spin()
