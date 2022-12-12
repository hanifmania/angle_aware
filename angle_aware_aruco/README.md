# angle_aware_aruco
土方さん農場での実験1. ArUcoマーカーとAngleAwareの実験

## Demo
Please click and check the Youtube.
[![](https://img.youtube.com/vi/FaWPN4Oym5M/0.jpg)](https://www.youtube.com/watch?v=FaWPN4Oym5M)


## Requirements
- angle_aware_control
- bebop_aruco


## Run test
### Simulation
```
roslaunch angle_aware_aruco bringup_sim.launch
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push X button -> main control start
1. Push START button -> take off and field reset


### Experiment
```
roslaungh angle_aware_aruco bringup.launch
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push START button -> takeoff
1. Push A button -> message control mode
1. Push X button -> main control start
1. Push A button -> joy_msg control mode (stop)
1. Push BACK button -> land


## Author

[Takumi Shimizu](https://github.com/tashiwater)

