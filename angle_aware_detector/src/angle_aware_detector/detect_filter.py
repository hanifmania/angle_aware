#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.agent_base import AgentBase

import rospy
from geometry_msgs.msg import PoseStamped, PoseArray

import numpy as np


class DetectFilter:
    def __init__(self):
        input_topic = rospy.get_param(
            "~input_topic", default="object_detector/posestamped"
        )
        output_topic = rospy.get_param(
            "~output_topic", default="object_detector/target_posestamped"
        )
        input_found_object = rospy.get_param(
            "~input_found_object", default="/found_objects"
        )
        self._world_tf = rospy.get_param("~world", default="world")
        grape_detector = rospy.get_param("/detect_filter")
        self._max_stock = rospy.get_param("/detect_filter/max_stock", 10)
        self._decide_count = grape_detector["decide_count"]
        self._object_span = grape_detector["object_span"]
        self._same_object_error = grape_detector["same_object_error"]
        self._range = grape_detector["range"]
        self._found_object = []
        self._object_candidate = []
        self._my_found_object = np.empty((1, 3))
        self._pub = rospy.Publisher(output_topic, PoseStamped, queue_size=1)
        rospy.Subscriber(
            input_topic, PoseStamped, self.detect_posestamped_callbeck, queue_size=5
        )
        rospy.Subscriber(
            input_found_object, PoseArray, self.found_object_callback, queue_size=1
        )

    #############################################################
    # callback
    #############################################################
    def detect_posestamped_callbeck(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        position = np.array([x, y, z])
        self.main(position)

    def found_object_callback(self, msg):
        positions = []
        orientations = []
        for pose in msg.poses:
            pos = [
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ]
            positions.append(pos)
            # q = [
            #     pose.orientation.x,
            #     pose.orientation.y,
            #     pose.orientation.z,
            #     pose.orientation.w,
            # ]
            # orientations.append(q)
        self._found_object = np.array(positions)
        # self._all_orientation = np.array(orientations)

    #############################################################
    # publish
    #############################################################
    def publish(self, position):
        """物体位置をpublish

        Args:
            position (ndarray): [x, y, z]
        """
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self._world_tf
        msg.pose.position.x = position[0]
        msg.pose.position.y = position[1]
        msg.pose.position.z = position[2]  # [TODO] get in param
        msg.pose.orientation.w = 1
        self._pub.publish(msg)

    #############################################################
    # main
    #############################################################

    def main(self, position):
        if not self.in_field(position, self._range):
            return
        already_detected = self.already_detected(
            position, self._found_object, self._object_span
        )
        if already_detected:
            return
        already_detected = self.already_detected(
            position, self._my_found_object, self._object_span
        )
        if already_detected:
            return
        result, self._object_candidate = self.double_check(
            position,
            self._object_candidate,
            self._same_object_error,
            self._decide_count,
            self._max_stock
        )
        if result is not False:
            self._my_found_object = np.vstack([self._my_found_object, result])
            self.publish(result)

    #############################################################
    # functions
    #############################################################

    def in_field(self, position, range):
        """検索範囲内にあるか確認

        Args:
            position (list): new object position xyz
            range (list): [[x_min, x_max], ...]

        Returns:
            bool: in field -> True
        """
        x, y, z  = position
        return (range[0][0] < x < range[0][1] ) and (range[1][0] < y < range[1][1] ) and  (range[2][0] < z < range[2][1] )

    def already_detected(self, grape_position, all_grape, object_span):
        """既知のものかチェック

        Args:
            grape_position (list): new grape position xyz
            all_grape (ndarray): 既知のぶどうリスト
            object_span (float): ぶどう間隔。この間隔以内のぶどうは同一とみなす

        Returns:
            bool: ぶどうが既知ならTrue
        """
        if len(all_grape) == 0:
            return False
        position = np.array(grape_position)
        dist = np.linalg.norm(position - all_grape, axis=1)
        min_dist = np.min(dist, axis=0)
        return min_dist < object_span

    def double_check(self, position, object_candidates, same_object_error, count, max_stock):
        """外れ値除去。count回同じ場所で検出したら初めて認められる

        Args:
            position (ndarray): [x, y, z]
            object_candidates (ndarray): list of postion
            same_object_error (float): 同じ物体とみなすしきい値
            count (int): 何回同じ場所で検出すべきか
            max_stock (int) : candidatesの最大個数

        Returns:
            ndarray: publish可能なら[x,y,z]. 不可能ならFalse
            ndarray: object_candidates
        """
        position = np.array(position)
        if len(object_candidates) == 0:
            object_candidates = position.reshape(1, -1)
            return False, object_candidates
        dist = np.linalg.norm(position - object_candidates, axis=1)
        same = dist < self._same_object_error
        ret = np.sum(same) >= count
        if ret:
            rospy.loginfo("before")
            rospy.loginfo(len(object_candidates))
            result = np.average(np.vstack([object_candidates[same], position]), axis=0)
            object_candidates = object_candidates[~same]
            rospy.loginfo("after")
            rospy.loginfo(len(object_candidates))
        else:
            object_candidates = np.vstack([object_candidates, position])
            result = False
        if len(object_candidates) >= max_stock:
            np.delete(object_candidates, 0,0)
        return result, object_candidates


if __name__ == "__main__":
    rospy.init_node("DetectFilter", anonymous=True)
    node = DetectFilter()
    rospy.spin()
