#!/usr/bin/env python
# -*- coding: utf-8 -*-
import roslib

roslib.load_manifest("diagnostic_updater")

import rospy
import diagnostic_updater

from std_msgs.msg import Bool
from diagnostic_msgs.msg import DiagnosticStatus


class PubDiagnostic:
    def __init__(self):
        ################## param input
        input_topic = rospy.get_param("~input_topic", default="angle_aware_mode")
        output_topic = rospy.get_param("~output_topic", default="mode")
        ##################

        self._updater = diagnostic_updater.Updater()
        # rospy.loginfo(rospy.get_namespace())
        self._updater.setHardwareID("mode")
        self._updater.add(output_topic, self.check)
        self._is_angle_aware_mode = False

        rospy.Subscriber(input_topic, Bool, self.callback, queue_size=1)

    #############################################################
    # callback
    #############################################################

    def callback(self, msg):
        self._is_angle_aware_mode = msg.data
        self._updater.force_update()

    #############################################################
    # main function
    #############################################################
    def check(self, stat):
        if self._is_angle_aware_mode:
            stat.summary(DiagnosticStatus.WARN, "AngleAware")
        else:
            stat.summary(DiagnosticStatus.OK, "Patrol")
        return stat

    #############################################################
    # main function
    #############################################################
    def spin(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self._updater.update()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("template", anonymous=True)
    node = PubDiagnostic()
    node.spin()
