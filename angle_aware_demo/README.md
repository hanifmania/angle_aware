# angle_aware_demo
This is the pkg for demo


## Demo

## Requirements
- angle_aware_switch

## Run Test
### Simulation
```
roslaunch angle_aware_demo bringup_sim.launch
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push X button -> main control start
1. Push START button -> take off and field reset

### Experiment
#### sky (Note that sky is so small that only 2 drone can work on this version)
- Set some chairs in [[-1,1], [-1,1]] field

##### Desktop PC (maxwell)
```
roslaunch angle_aware_demo central.launch agentNum:="2"
roslaunch angle_aware_demo agent.launch number:="1"
```

##### Laptop PC
```
roslaunch angle_aware_demo agent.launch number:="2"
```

## Rosbag -> csv
```
roscd angle_aware_demo/rosbag/
rostopic echo -b rosbag.bag -p /patrol_J > J.csv
rostopic echo -b rosbag.bag -p /bebop101/angle_aware/J > angle_aware_J1.csv
rostopic echo -b rosbag.bag -p /bebop102/angle_aware/J > angle_aware_J2.csv
```

## Author

[Takumi Shimizu](https://github.com/tashiwater)

