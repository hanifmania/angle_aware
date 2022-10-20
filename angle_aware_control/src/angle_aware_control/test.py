#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped

import numpy as np
import copy
import numpy as np
from angle_aware_control.numpy2multiarray import numpy2multiarray
from std_msgs.msg import Float32MultiArray

# x = np.array([0,1,2])
# y = np.array([2,4])
# z = np.array([2])
# phi = np.ones((3, 2))
# numpy2multiarray(Float32MultiArray , phi)
# grid = np.meshgrid(x,y, indexing="ij", sparse=True)
# print(phi)
# print(grid)
# pos = np.array([5,3])
# print(phi * grid[0])