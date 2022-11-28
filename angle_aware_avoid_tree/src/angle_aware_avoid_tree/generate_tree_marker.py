#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3, Pose
from std_msgs.msg import ColorRGBA


class GenerateTreeMarker:
    def __init__(self):
        world_tf = rospy.get_param("~world", default="world")
        output_topic = rospy.get_param("~output_topic", default="tree_markers")
        color = rospy.get_param("~color", default=[1, 1, 1, 0.8])
        height = rospy.get_param("agents/ref_z")
        tree_param = rospy.get_param("trees")
        tree_positions = tree_param["xy"]
        tree_radius = tree_param["avoid_radius"]

        self._pub = rospy.Publisher(output_topic, MarkerArray, queue_size=1)

        msg = self.generate_marker_array(
            tree_positions, tree_radius, height, color, world_tf
        )
        while not rospy.is_shutdown():
            rospy.sleep(1)
            self._pub.publish(msg)

    #############################################################
    # functions
    #############################################################

    def tree_positions_to_poses(self, tree_positions):
        """xyからposeをつくる

        Args:
            tree_positions (list): xy座標のlist

        Returns:
            list: Poseのlist
        """
        poses = []
        for xy in tree_positions:
            pose = Pose()
            pose.position.x = xy[0]
            pose.position.y = xy[1]
            pose.position.z = 0
            pose.orientation.w = 1
            poses.append(pose)
        return poses

    def tree_radius_to_scales(self, tree_radius, height):
        """半径からscaleをつくる

        Args:
            tree_radius (list): rのlist
            height(float): z

        Returns:
            list: Vector3 の list
        """
        scales = []
        for r in tree_radius:
            scale = Vector3()
            scale.x = r
            scale.y = r
            scale.z = height
            scales.append(scale)
        return scales

    def generate_cylinder(self, id, x, y, height, r, color, world_tf):
        """_summary_

        Args:
            id(int) : marker.id
            x (float): _description_
            y (float): _description_
            height (float): _description_
            r (float): _description_
            color (list): [r, g, b, a]
            world_tf (str): _description_

        Returns:
            Marker: _description_
        """
        marker = Marker()
        marker.header.frame_id = world_tf
        marker.header.stamp = rospy.Time.now()
        marker.id = id
        marker.lifetime = rospy.Duration()

        marker.action = Marker.ADD
        marker.type = Marker.CYLINDER

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = height * 0.5
        marker.pose.orientation.w = 1

        marker.scale.x = r
        marker.scale.y = r
        marker.scale.z = height
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    #############################################################
    # main
    #############################################################
    def generate_marker_array(
        self, tree_positions, tree_radius, height, color, world_tf
    ):
        """_summary_

        Args:
            tree_positions (list): _description_
            tree_radius (list): _description_
            height(float): z
            color (list): _description_
            world_tf (str): _description_

        Returns:
            MarkerArray: _description_
        """

        marker_array = MarkerArray()
        for i, (xy, r) in enumerate(zip(tree_positions, tree_radius)):
            marker = self.generate_cylinder(i, xy[0], xy[1], height, r, color, world_tf)
            marker_array.markers.append(marker)
        return marker_array

    #############################################################
    # old code
    #############################################################
    def generate_marker_from_url(self, id, pose, scale, model_url, world_tf):
        """dae, stl, glb形式のモデルを生成

        Args:
            id(int) : marker.id
            pose (Pose): _description_
            scale (Vector3): _description_
            model_url (str): _description_
            world_tf (str): _description_

        Returns:
            Marker: _description_
        """
        marker = Marker()
        marker.header.frame_id = world_tf
        marker.header.stamp = rospy.Time.now()
        marker.id = id
        marker.lifetime = rospy.Duration()

        marker.action = Marker.ADD
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_use_embedded_materials = True
        marker.mesh_resource = model_url

        marker.pose = pose
        marker.scale = scale
        return marker


if __name__ == "__main__":
    rospy.init_node("template", anonymous=True)
    node = GenerateTreeMarker()
