#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import PoseStamped

import numpy as np
import pandas as pd
import datetime


class LogError:
    def __init__(self):
        input_ar_topic = rospy.get_param("~input_ar_topic", "posestamped")
        input_vrpn_topic = rospy.get_param("~input_vrpn_topic")
        self._log_path = rospy.get_param("~log_path")

        self._vrpn = PoseStamped()
        self._start_t = rospy.Time.now()
        self._log = []
        rospy.Subscriber(input_ar_topic, PoseStamped, self.ar_callback)
        rospy.Subscriber(input_vrpn_topic, PoseStamped, self.vrpn_callback)
        rospy.on_shutdown(self.savelog)

    #############################################################
    # callback
    #############################################################
    def ar_callback(self, msg):
        ar_xyz = self.pose_to_xyz(msg.pose)
        vrpn_xyz = self.pose_to_xyz(self._vrpn.pose)
        norm = np.linalg.norm([ar_xyz - vrpn_xyz])
        t = msg.header.stamp - self._start_t
        t_sec = t.to_sec()
        data = np.hstack([t_sec, norm, ar_xyz, vrpn_xyz])
        self._log.append(data.tolist())

    def vrpn_callback(self, msg):
        self._vrpn = msg

    #############################################################
    # functions
    #############################################################

    def pose_to_xyz(self, pose):
        xyz = np.array([pose.position.x, pose.position.y, pose.position.z])
        return xyz
    
    def savelog(self, other_str = ""):
        now = datetime.datetime.now()
        filename = self._log_path+"/" + now.strftime("%Y%m%d_%H%M%S") + other_str + ".csv"
        df = pd.DataFrame(
            data=self._log,
            columns=[
                "time [s]",
                "norm",
                "ar_x",
                "ar_y",
                "ar_z",
                "vrpn_x", "vrpn_y", "vrpn_z"
                ],
        )
        df.to_csv(filename, index=True)
        rospy.loginfo("save " + filename)

    # def __delete__(self):
    #     self.savelog()


if __name__ == "__main__":
    rospy.init_node("template", anonymous=True)
    node = LogError()
    rospy.spin()