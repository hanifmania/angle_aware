#!/usr/bin/env python
# -*- coding: utf-8 -*-
from bebop_hatanaka_base.agent_base import AgentBase
from coverage_util.field_generator import FieldGenerator
from coverage_util.numpy2multiarray import multiarray2numpy
from coverage_util.voronoi import CoverageUtil

import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist

import numpy as np
import pandas as pd
import datetime

class Log:
    def __init__(self):
        self.agentID = rospy.get_param("agentNum")
        angle_aware_params = rospy.get_param("angle_aware")
        self._log_path = rospy.get_param("~log_path")
        target_id = rospy.get_param("~target_id", default=1)
        input_psi_topic = rospy.get_param("~input_psi_topic", default="psi")

        self._sigma = angle_aware_params["sigma"]
        bebop_name = "bebop10{}/".format(target_id)
        cmd_input_topic = bebop_name + "cmd_input"
        psi_param = angle_aware_params["psi"]
        self._pos_index = target_id - 1 
        psi_generator = FieldGenerator(psi_param)
        self._psi_grid = psi_generator.generate_grid()

        self._log = []
        self._start_t = None
        self._vel = Twist()

        self._agent_base = AgentBase()
        self._coverage_util = CoverageUtil()

        rospy.Subscriber(input_psi_topic, Float32MultiArray, self.psi_callback)
        rospy.Subscriber(cmd_input_topic, Twist, self.vel_callback)
        rospy.on_shutdown(self.savelog)

    #############################################################
    # callback
    #############################################################
    def psi_callback(self, msg):
        self._psi = multiarray2numpy(float, np.float32, msg)
    
    def vel_callback(self, msg):
        self._vel = msg
        self.add_log()

    ###################################################################
    ### main
    ###################################################################
    def savelog(self, other_str = ""):
        now = datetime.datetime.now()
        # filename = self._log_path+"/" + now.strftime("%Y%m%d_%H%M%S") + other_str + ".csv"
        filename = self._log_path
        df = pd.DataFrame(
            data=self._log,
            columns=[
                "time [s]",
                "J",
                "J_near",
                "|u|",
            ]
        )
        df.to_csv(filename, index=True)
        rospy.loginfo("save " + filename)

    def add_log(self):
        t = self.calc_t()
        J = self.calc_J()
        u = self.calc_vel()
        J_near = self.calc_J_near()
        temp = [t, J, J_near, u]
        self._log.append(temp)


    ###################################################################
    ### functions
    ###################################################################
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

    def calc_J(self):
        """重要度の総和

        Returns:
            float: 重要度の総和
        """        
        return np.sum(self._psi)
    
    def calc_vel(self):
        """|u|. [WARN] 機体の傾きを考慮していない。

        Returns:
            float: |u|
        """        
        vx = self._vel.linear.x
        vy = self._vel.linear.y
        return np.linalg.norm([vx, vy])

    def calc_J_near(self):
        """ 近傍の重要度

        Returns:
            float: J_near
        """
        all_positions = self._agent_base.get_all_positions()

        my_position = all_positions[self._pos_index]
        dist = self._coverage_util.calc_dist(my_position, self._psi_grid)
        return np.sum((dist < 2 * self._sigma) * self._psi)



if __name__ == "__main__":
    rospy.init_node("log", anonymous=True)
    node = Log()
    rospy.spin()
