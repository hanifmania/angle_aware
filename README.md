# AngleAware2
New Angle Aware.  
This is the code for https://ieeexplore.ieee.org/document/9650560.
This program uses https://github.com/htnk-lab/bebop_hatanaka_lab.git

# Contents
- angle_aware_control :
    - agent.py :  圧縮した重要度psiを受け取ってangle aware CBFに従いドローンを操作する
        - angle_aware_cbf.py : Angle Aware CBFのアルゴリズム
        - myqp.py : collision avoidance 等も踏まえた最終的なQP
        - coverage_util.py : Voronoi
    - central.py : 圧縮した重要度psiを更新し、publish
    - field_generator.py : 重要度mapのgrid生成
    - show_pointcloud2d.py : psiをrvizに描写

# Dependency
## Environment
### Simulation and Experiment
- ubuntu16.04
- ros kinetic
### Generate Psi and Simulation
- ubuntu20.04
- ros noetic

## Install
- git clone https://github.com/htnk-lab/bebop_hatanaka_lab.git
- git clone https://github.com/htnk-lab/persistent_coverage_control.git
- Follow the README of https://github.com/htnk-lab/bebop_hatanaka_lab.git

# Build
```
catkin build
```

# Demo
## Simulation
```
roslaunch angle_aware_control bringup_sim.launch
```
1. Set Logicool joy stick with Xbox mode and mode light off.
1. Push X button -> main control start
1. Push START button -> take off and field reset

## Experiment
```
roslaunch angle_aware_control central.launch agentNum:="3"
roslaunch angle_aware_control agent.launch number:=""
```

## Generate Psi
- Change numpy to jax.numpy in angle_aware_control/src/field_generator.py Line.3
```
import numpy as np -> import jax.numpy as np
```
- launch
```
roslaunch angle_aware_control generate_psi.launch
```

