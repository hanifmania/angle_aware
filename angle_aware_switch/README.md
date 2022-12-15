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
### Generate Psi
```
roslaunch angle_aware_control patrol_generate_psi.launch
```
### Simulation
```
roslaunch angle_aware_switch bringup_sim.launch
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push X button -> main control start
1. Push START button -> take off and field reset

### Experiment
#### 2 drone version
Connect 2 PC with LAN cable.
##### PC 1
- Set the private IP (192.168.1.1, netmask 255.255.255.0, gateway:192.168.1.2)
- gedit ~/.bashrc
```
# export ROS_IP=127.0.0.1
# export ROS_MASTER_URI=http://localhost:11311
export ROS_IP=$(hostname -I | cut -d' ' -f1)
export ROS_MASTER_URI=http://$ROS_IP:11311
```
- Open new terminal and launch
```
roslaunch angle_aware_switch master.launch
```
##### PC 2
- Set the private IP (192.168.1.2, netmask 255.255.255.0, gateway:192.168.1.1)
- gedit ~/.bashrc
```
export ROS_IP=$(hostname -I | cut -d' ' -f1)
export ROS_MASTER_URI=http://192.168.1.111311
```
- Open new terminal and launch
```
roslaunch angle_aware_switch agent.launch number:="2" use_aruco:="true"
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push START button -> takeoff
1. Push A button -> message control mode
1. Push X button -> main control start
1. Push A button -> joy_msg control mode (stop)
1. Push BACK button -> land

## Author

[Takumi Shimizu](https://github.com/tashiwater)

