#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings("ignore")  ##ignore the warning due to the python2

from bebop_hatanaka_base import ros_utility
from bebop_aruco.ar_detect import ARDetect
from coverage_util.field_generator import FieldGenerator

import tf

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PoseStamped, Pose

import numpy as np

import pandas as pd

# from bebop_aruco.ar_detect import ARDetect


class CameraFromAR:
    def __init__(self):
        # input_img_topic = rospy.get_param("~input_img_topic", "image_raw")
        input_img_compressed_topic = rospy.get_param(
            "~input_img_compressed_topic", "image_raw/compressed"
        )
        camera_matrix_param = rospy.get_param("~camera_matrix")
        D_param = rospy.get_param("~distortion_coefficients")
        # pose_stamped_feedback = rospy.get_param("~pose_stamped_feedback", "posestamped")
        output_pose_topic = rospy.get_param(
            "~output_pose_topic", "fromAR/camera_posestamped"
        )
        self._frame_id = rospy.get_param("~frame_id", "world")
        ar_params = rospy.get_param("/AR")
        self._ar_marker_size = ar_params["ar_marker_size"]  # ARマーカー一辺. 黒い外枠含む[m]
        self._ar_type = ar_params[
            "ar_type"
        ]  # AR markerの種類(https://docs.opencv.org/3.4/d9/d6a/group__aruco.html#gac84398a9ed9dd01306592dd616c2c975)
        ar_field_param = ar_params["map"]  # IDは連番で、左下のID=0となるよう配置する
        # self._time_threshold = ar_params["time_threshold"]
        # self._norm_threshold = ar_params["norm_threshold"]
        self._id_offset = ar_params["id_offset"]
        self._outer_velocity = ar_params["outer_velocity"]
        self._stop_skip_t = ar_params["stop_skip_t"]

        self._map_generator = FieldGenerator(ar_field_param)
        self._grid_x, self._grid_y = self._map_generator.generate_grid(sparse=False)
        field_shape = self._map_generator.get_shape()
        self._max_marker_id = field_shape.prod() - 1 + self._id_offset
        self._cv_bridge = CvBridge()
        self._ar = ARDetect()
        # self._ar.set_params(self._ar_marker_size, self._ar_type,None, None)

        # self._has_camera_info_received = False
        self._pose_stamped_feedback = PoseStamped()
        self._old_xyz = None
        self._old_quaternion = None
        self._old_t = None
        self._last_skip_t = rospy.Time.now()

        K = camera_matrix_param["data"]
        D = D_param["data"]
        self._camera_matrix = np.array(K).reshape(3, -1)
        self._dist_coeffs = np.array(D)
        self._ar.set_params(
            self._ar_marker_size, self._ar_type, self._camera_matrix, self._dist_coeffs
        )

        self._pub_posestamped = rospy.Publisher(
            output_pose_topic, PoseStamped, queue_size=1
        )
        # rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(input_img_compressed_topic, CompressedImage, self.img_callback)
        # rospy.Subscriber(input_img_topic, Image, self.img_callback)
        # rospy.Subscriber(pose_stamped_feedback, PoseStamped, self.pose_stamped_feedback)

        # density = ar_field_param["density"]

        # self._ar.set_borad(field_shape[0], field_shape[1], density[0])
        self._log_path = rospy.get_param("~log_path", default="")
        self._log = []
        self._start_t = None

        # rospy.on_shutdown(self.savelog)

    #############################################################
    # callback
    #############################################################
    def img_callback(self, img_msg):
        """image_rawからカメラ位置を推定する

        Args:
            img_msg (sensor_msgs.msg.Image): _description_
        """
        # if self._has_camera_info_received is False:
        #     ### camera info がなければearly return
        #     return
        ids = self.find_marker(img_msg)
        # rospy.loginfo("AR marker")
        # rospy.loginfo(ids)
        if ids is None or min(ids) > self._max_marker_id:
            ### AR markerが見つからなければearly return
            return

        ### 画像から読み取った camera位置の候補
        xyzs, Rs = self.sampling_pose(ids)
        ### 外れ値の除去
        # filterd_xyzs, filtered_Rs =  self.outlier_filter(xyzs, Rs)
        # if len(filterd_xyzs) == 0:
        #     return
        ### camera位置の推定

        filtered_xyzs, filtered_Rs = self.outer_filter(xyzs, Rs)
        if len(filtered_xyzs) == 0:
            rospy.loginfo("skip all")
            return
        xyz, R = self.estimate_pose(filtered_xyzs, filtered_Rs)
        ### cv2の座標系からbebopの座標系に変換
        bebop_R = self.change_axis(R)

        ### 結果のpublish
        posestamped = self.np2posestamped(xyz, bebop_R, img_msg)
        self._pub_posestamped.publish(posestamped)

    # def camera_info_callback(self, msg):
    #     """camera行列等の取得. main_function実行前に実施する必要がある

    #     Args:
    #         msg (sensor_msgs.msg.CameraInfo): _description_
    #     """
    #     # if self._has_camera_info_received:
    #     #     return
    #     self._camera_matrix = np.array(msg.K).reshape(3, -1)
    #     self._dist_coeffs = np.array(msg.D)
    #     self._ar.set_params(self._ar_marker_size, self._ar_type, self._camera_matrix, self._dist_coeffs)

    #     self._has_camera_info_received =True

    def pose_stamped_feedback(self, msg):
        self._pose_stamped_feedback = msg

    #############################################################
    # functions
    #############################################################

    def find_marker(self, img_msg):
        """ARマーカーの検出

        Args:
            img_msg (sensor_msgs/CompressedImage): 入力画像

        Returns:
            list: 検出されたマーカーIDのlist
        """
        # cv_array = self._cv_bridge.imgmsg_to_cv2(img_msg)
        cv_array = self._cv_bridge.compressed_imgmsg_to_cv2(img_msg)
        self._ar.set_img(cv_array)
        within_eyesight_marker_ids = self._ar.find_marker()
        return within_eyesight_marker_ids

    def sampling_pose(self, ids):
        """画像からカメラ姿勢の検出

        Args:
            ids (list): 検出されたマーカーのid list

        Returns:
            ndarray: カメラ位置のlist
            ndarray: カメラ姿勢のlist
        """
        xyzs = []
        Rs = []
        ids = ids.reshape(-1)
        for id in ids:
            ### 用意していないMarkerをご検知したらskip
            if id > self._max_marker_id:
                continue
            ### ARマーカーを原点としたカメラの位置を取得
            xyz_from_ar, R = self._ar.get_camera_pose(id)

            ### AR mapを用いて、world座標系に変換
            map_index = self._map_generator.sequence2index(id - self._id_offset)
            x = xyz_from_ar[0] + self._grid_x[map_index[0], map_index[1]]
            y = xyz_from_ar[1] + self._grid_y[map_index[0], map_index[1]]
            z = xyz_from_ar[2]
            xyz = np.array([x, y, z])
            xyzs.append(xyz)
            Rs.append(R)

        return np.array(xyzs), np.array(Rs)

    def estimate_pose(self, xyzs, Rs):
        """samplingした結果から代表点を生成

        Args:
            xyzs (ndarray): xyzのlist
            Rs (ndarray): 回転行列のlist

        Returns:
            ndarray: xyzの平均を
            ndarray: 回転行列の平均
        """

        ### xyzはsimpleに平均を取る
        average_xyz = np.mean(xyzs, axis=0)
        ### 角度は定義域があるため単純な平均が出来ない(例：0度と360度の平均が180度になってしまう)
        ### [TODO] quaternionを使って平均を取る
        average_R = self.average_R(Rs)

        ### low pass filter
        if self._old_xyz is not None:
            average_xyz = np.mean([average_xyz, self._old_xyz], axis=0)
            average_R = self.average_R([self._old_R, average_R])
        self._old_xyz = average_xyz
        self._old_R = average_R

        return average_xyz, average_R

    def average_R(self, Rs):
        """固有値を用いて平均を求める。参考： https://qiita.com/qoopen0815/items/d05e49dd4ce7c1f0c524

        Args:
            Rs (list): 回転行列のlist

        Returns:
            ndarray: 平均回転行列
        """
        ###
        # rospy.loginfo("quaternion")
        q_row = np.empty((len(Rs), 4))
        for i, R in enumerate(Rs):
            quaternion = ros_utility.R_to_quaternion(R)
            q_row[i] = quaternion
            # rospy.loginfo(quaternion)
        M = q_row.T.dot(q_row)
        values, vectors = np.linalg.eig(M)
        max_eigen_index = np.argmax(values)
        # rospy.loginfo("average")
        average_quaternion = vectors[:, max_eigen_index].reshape(-1)
        # rospy.loginfo(average_quaternion)
        g = tf.transformations.quaternion_matrix(average_quaternion)
        R = g[:3, :3]
        return R

    def outer_filter(self, xyzs, Rs):
        now = rospy.Time.now()
        if self._old_t is None:
            self._old_t = now
            return xyzs, Rs

        dt = now - self._old_t
        self._old_t = now

        safe_index = []
        for i, xyz in enumerate(xyzs):
            vel = np.linalg.norm(xyz - self._old_xyz) / dt.to_sec()
            if vel < self._outer_velocity:
                safe_index.append(i)
            else:
                rospy.loginfo("outlier : {}".format(vel))
        filtered_xyzs, filtered_Rs = xyzs[safe_index], Rs[safe_index]

        if len(filtered_xyzs) == 0:
            ### dataが全部外れ値だった場合、それが長時間起こると現在地が更新されず危険。
            skip_t = now - self._last_skip_t
            if skip_t.to_sec() > self._stop_skip_t:
                ### 一定時間以上更新しなければ外れ値除去を一度やめる
                filtered_xyzs, filtered_Rs = xyzs, Rs
                self._last_skip_t = now
        else:
            ### skipしなれけば、last skip timeを更新
            self._last_skip_t = now
        return filtered_xyzs, filtered_Rs

    def change_axis(self, R):
        """cvのカメラ座標をbebop2droneの座標系に変換

        Args:
            R (ndarray): cvが想定するカメラの回転行列

        Returns:
            ndarray: bebop2が想定するカメラの回転行列
        """
        # z軸周りに回転
        z = np.pi / 2
        z_rotate = np.array(
            [[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]]
        )
        R2 = R.dot(z_rotate)
        # y軸周りに回転
        y = -np.pi / 2
        y_rotate = np.array(
            [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
        )
        bebop_R = R2.dot(y_rotate)
        return bebop_R

    def np2posestamped(self, xyz, R, img_msg):
        """ndarrayからposestampedを生成

        Args:
            xyz (ndarray): カメラ位置
            R (ndarray): 回転行列
            img_msg (sensor_msgs/Image): header取得用

        Returns:
            geometry_msgs/PoseStamped: カメラ姿勢
        """
        # quaternion = tf.transformations.quaternion_from_euler(*rpy)

        quaternion = ros_utility.R_to_quaternion(R)
        posestamped = PoseStamped()
        posestamped.header = img_msg.header
        posestamped.header.frame_id = self._frame_id
        posestamped.pose.position.x = xyz[0]
        posestamped.pose.position.y = xyz[1]
        posestamped.pose.position.z = xyz[2]
        posestamped.pose.orientation.x = quaternion[0]
        posestamped.pose.orientation.y = quaternion[1]
        posestamped.pose.orientation.z = quaternion[2]
        posestamped.pose.orientation.w = quaternion[3]

        return posestamped

    ####################
    def add_log(self, xyzs, old_xyz):
        if old_xyz is None:
            return
        t = self.calc_t()
        for xyz in xyzs:
            temp = [
                t,
                xyz[0],
                xyz[1],
                xyz[2],
                old_xyz[0],
                old_xyz[1],
                old_xyz[2],
            ]
            self._log.append(temp)

    def calc_t(self):
        """最初のlog移項の時刻をカウントする.

        Returns:
            float: time [s]
        """
        t = rospy.Time.now()
        if self._start_t is None:
            self._start_t = t
        t2 = t - self._start_t
        return t2.to_sec()

    def savelog(self, other_str=""):
        # now = datetime.datetime.now()
        # filename = self._log_path+"/" + now.strftime("%Y%m%d_%H%M%S") + other_str + ".csv"
        filename = self._log_path
        df = pd.DataFrame(
            data=self._log,
            columns=[
                "time [s]",
                "x",
                "y",
                "z",
                "old_x",
                "old_y",
                "old_z",
            ],
        )
        df.to_csv(filename, index=False)
        rospy.loginfo("save " + filename)

    # def outlier_filter(self, xyzs, Rs):
    #     """feedbackを用いて外れ値を除去する. 位置が大きく前回とずれていたら外れ値と判断する.

    #     Args:
    #         xyzs (ndarray): xyz list
    #         Rs (ndarray): 回転行列 list

    #     Returns:
    #         ndarray: 外れ値を削除したxyz list
    #         ndarray: 外れ値を削除した回転行列 list
    #     """
    #     now = rospy.Time.now()
    #     diff = now - self._pose_stamped_feedback.header.stamp
    #     if diff.to_sec() > self._time_threshold:
    #         ### feedbackがかなり前のものなら外れ値検知として使えない
    #         return xyzs, Rs

    #     feedback_xyz = np.array([self._pose_stamped_feedback.pose.position.x,
    #                             self._pose_stamped_feedback.pose.position.y,
    #                             self._pose_stamped_feedback.pose.position.z])

    #     diff_xyzs = xyzs - feedback_xyz
    #     norms =  np.linalg.norm(diff_xyzs, axis=1)
    #     ok = norms < self._norm_threshold
    #     rospy.loginfo(norms)
    #     rospy.loginfo(ok)
    #     removed_xyzs = xyzs[ok]
    #     removed_Rs = Rs[ok]

    #     return removed_xyzs, removed_Rs


if __name__ == "__main__":
    rospy.init_node("ar2camera", anonymous=True)
    node = CameraFromAR()
    rospy.spin()
