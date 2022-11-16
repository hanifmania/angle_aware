#!/usr/bin/env python
# -*- coding: utf-8 -*-
from angle_aware_control.agent import Agent
from angle_aware_avoid_tree.myqp import QPAvoidTree
import rospy

if __name__ == "__main__":
    rospy.init_node("agent", anonymous=True)
    trees = rospy.get_param("/trees")
    agent = Agent(QPAvoidTree)
    agent._qp.set_params(trees)
    agent.spin()
