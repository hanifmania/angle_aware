#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3, Pose


class GenerateTreeMarker:
    def __init__(self):
        model_url = rospy.get_param("~model_url")
        world_tf = rospy.get_param("~world", default="world")
        output_topic = rospy.get_param("~output_topic", default="tree_markers")
        tree_param = rospy.get_param("trees")
        tree_positions = tree_param["xy"]
        tree_radius = tree_param["avoid_radius"]

        self._pub = rospy.Publisher(output_topic, MarkerArray, queue_size=1)

        poses = self.tree_positions_to_poses(tree_positions)
        scales = self.tree_radius_to_scales(tree_radius)
        msg = self.generate_markers(poses, scales, model_url, world_tf)
        while not rospy.is_shutdown():
            rospy.sleep(1)
            self._pub.publish(msg)

    #############################################################
    # functions
    #############################################################

    def tree_positions_to_poses(self, tree_positions):
        poses = []
        for xy in tree_positions:
            pose = Pose()
            pose.position.x = xy[0]
            pose.position.y = xy[1]
            pose.position.z = 0
            pose.orientation.w = 1
            poses.append(pose)
        return poses

    def tree_radius_to_scales(self, tree_radius):
        scales = []
        for r in tree_radius:
            scale = Vector3()
            scale.x = r
            scale.y = r
            scale.z = r
            scales.append(scale)
        return scales

    #############################################################
    # main
    #############################################################

    def generate_markers(self, poses, scales, model_url, world_tf):
        marker_array = MarkerArray()
        for i, (pose, scale) in enumerate(zip(poses, scales)):
            marker = Marker()
            marker.header.frame_id = world_tf
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.lifetime = rospy.Duration()

            marker.action = Marker.ADD
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_use_embedded_materials = True
            marker.mesh_resource = model_url

            marker.pose = pose
            marker.scale = scale
            marker_array.markers.append(marker)
        return marker_array


if __name__ == "__main__":
    rospy.init_node("template", anonymous=True)
    node = GenerateTreeMarker()
