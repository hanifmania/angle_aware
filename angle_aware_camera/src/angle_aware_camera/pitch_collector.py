#!/usr/bin/env python
# -*- coding: utf-8 -*-
from coverage_util.numpy2multiarray import numpy2multiarray

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


import numpy as np


class Pitch:
    def __init__(self, agentName):
        input_topic = rospy.get_param("~input_topic")
        topic_name = agentName + input_topic
        rospy.Subscriber(topic_name, Twist, self.callback)
        self._msg = Twist()

    #############################################################
    # callback
    #############################################################

    def callback(self, msg):
        self._msg = msg

    #############################################################
    # functions
    #############################################################
    def get_pitch(self):
        return self._msg.angular.y


class PitchCollector:
    def __init__(self):
        self._lock = rospy.get_param("~clock")
        self._agent_num = rospy.get_param("agentNum")
        output_topic = rospy.get_param("~output_topic")
        self._pitch_holders = [
            Pitch("bebop10{}/".format(i + 1)) for i in range(self._agent_num)
        ]

        self._pub = rospy.Publisher(output_topic, Float32MultiArray, queue_size=1)

    #############################################################
    # functions
    #############################################################

    def publish(self):
        pitches = np.zeros(self._gent_num)
        for i, pitch_holder in enumerate(self._pitch_holders):
            val = pitch_holder.get_pitch()
            pitches[i] = val

        msg = numpy2multiarray(pitches)
        self._pub.publish(msg)

    #############################################################
    # spin
    #############################################################

    def spin(self):
        rate = rospy.Rate(self._clock)
        while not rospy.is_shutdown():
            self.publish()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("PitchCollector", anonymous=True)
    node = PitchCollector()
    rospy.spin()
