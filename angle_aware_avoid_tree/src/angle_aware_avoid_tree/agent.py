#!/usr/bin/env python
# -*- coding: utf-8 -*-
from angle_aware_control.agent_with_unom import Agent
from angle_aware_control.myqp import MyQP
import rospy

if __name__ == "__main__":
    rospy.init_node("agent", anonymous=True)
    trees = rospy.get_param("/trees")
    agent = Agent(MyQP)
    agent._qp.set_obstacle_avoidance_param(trees)
    agent.spin()
