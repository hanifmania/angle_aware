# angle_aware_switch

## Demo
Please click and check the Youtube.   
[![](https://img.youtube.com/vi/LOIGUzbYuJA/0.jpg)](https://www.youtube.com/watch?v=LOIGUzbYuJA)

## Requirements
- angle_aware_control
- angle_aware_avoid_tree
- angle_aware_detector
- angle_aware_aruco

## Run Test
### Set config file
Set parameters from ..._xxx.yaml files (e.g., angle_aware_sky.yaml)
``` 
roscd angle_aware_switch
bash script/set_config.sh xxx
```

### Generate Psi
```
roslaunch angle_aware_switch patrol_generate_psi.launch
```


### Simulation
```
roslaunch angle_aware_switch bringup_sim.launch
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push X button -> main control start
1. Push START button -> take off and field reset

### Experiment
#### sky 
- Note that sky is so small that only 2 drone can work on this version
- Set an chair around the center of the sky
```
roslaunch angle_aware_switch central.launch agentNum:="2"
roslaunch angle_aware_switch agent.launch number:="1"
roslaunch angle_aware_switch agent.launch number:="2" ### on laptop computer
```

#### Hijikata Agri demo
Connect 2 PC with LAN cable.
##### PC 1
- Set the IP  by PC settings as Address 192.168.208.12, netmask 255.255.255.0, gateway:192.168.208.32
- gedit ~/.bashrc
```
# export ROS_IP=127.0.0.1
# export ROS_MASTER_URI=http://localhost:11311
export ROS_IP=192.168.208.12
export ROS_MASTER_URI=http://$ROS_IP:11311
```
- Open new terminal and launch
```
roslaunch angle_aware_switch orchard_master.launch
```
##### PC 2
```
roslaunch angle_aware_switch orchard_agent.launch
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push START button -> takeoff
1. Push A button -> message control mode
1. Push X button -> main control start
1. Push A button -> joy_msg control mode (stop)
1. Push BACK button -> land



## Rosbag -> csv
```
roscd angle_aware_switch/rosbag/
rostopic echo -b rosbag.bag -p /patrol_J > J.csv
rostopic echo -b rosbag.bag -p /bebop101/angle_aware/J > angle_aware_J1.csv
rostopic echo -b rosbag.bag -p /bebop102/angle_aware/J > angle_aware_J2.csv
```

## Author

[Takumi Shimizu](https://github.com/tashiwater)

