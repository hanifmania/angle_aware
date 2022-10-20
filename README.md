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
- ubuntu16.04
- ros kinetic

## Install
- git clone https://github.com/htnk-lab/bebop_hatanaka_lab.git
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


