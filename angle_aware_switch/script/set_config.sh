#!/bin/bash

source ${ROS_ROOT}/../rosbash/rosbash
roscd angle_aware_switch/config
cp central_$1.yaml central.yaml
cp angle_aware_$1.yaml angle_aware.yaml
cp AR_$1.yaml AR.yaml
cp object_detector_$1.yaml object_detector.yaml
cp patrol_$1.yaml patrol.yaml