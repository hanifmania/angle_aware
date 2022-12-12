# angle_aware_switch

<!-- ## Demo -->

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
- Set the private IP (192.168.1.1, gateway:192.168.1.2)
```
roslaunch angle_aware_switch master.launch
```
##### PC 2
- Set the private IP (192.168.1.2, gateway:192.168.1.1)
```
roslaunch angle_aware_switch agent.launch number:="2"
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push START button -> takeoff
1. Push A button -> message control mode
1. Push X button -> main control start
1. Push A button -> joy_msg control mode (stop)
1. Push BACK button -> land

## Author

[Takumi Shimizu](https://github.com/tashiwater)

