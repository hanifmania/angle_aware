# angle_aware_avoid_tree
Angle Aware with Camera control. Note that this is the simple version.

## Demo
Please click and check the Youtube video.  
[![](https://img.youtube.com/vi/QJO6noQm628/0.jpg)](https://www.youtube.com/watch?v=QJO6noQm628)

## Requirements
- angle_aware_control

## Run test
### Simulation
```
roslaunch angle_aware_camera bringup_sim.launch
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push X button -> main control start
1. Push START button -> take off and field reset


### Experiment (Not tested)
```
roslaunch angle_aware_camera central.launch agentNum:="3"
roslaunch angle_aware_camera agent.launch number:=""
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push START button -> takeoff
1. Push A button -> message control mode
1. Push X button -> main control start
1. Push A button -> joy_msg control mode (stop)
1. Push BACK button -> land


## Author

[Takumi Shimizu](https://github.com/tashiwater)

