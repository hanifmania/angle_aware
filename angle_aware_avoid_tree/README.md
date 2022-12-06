# angle_aware_avoid_tree
Add object avoidance to angle_aware_control

## Requirements
- angle_aware_control

## Run test
### Simulation
```
roslaunch angle_aware_avoid_tree bringup_sim.launch
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push X button -> main control start
1. Push START button -> take off and field reset


### Experiment
```
roslaunch angle_aware_avoid_tree central.launch agentNum:="3"
roslaunch angle_aware_avoid_tree agent.launch number:=""
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push START button -> takeoff
1. Push A button -> message control mode
1. Push X button -> main control start
1. Push A button -> joy_msg control mode (stop)
1. Push BACK button -> land


## Author

[Takumi Shimizu](https://github.com/tashiwater)

