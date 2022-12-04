# AngleAware2
New Angle Aware.  
This is the code for https://ieeexplore.ieee.org/document/9650560.
This program uses https://github.com/htnk-lab/bebop_hatanaka_lab.git

# Contents
- angle_aware_control :
    - agent.py :  圧縮した重要度psiを受け取ってangle aware CBFに従いドローンを操作する
    - angle_aware_cbf.py : Angle Aware CBFのアルゴリズム
    - myqp.py : collision avoidance 等も踏まえた最終的なQP
    - central.py : 圧縮した重要度psiを更新し、publish
    - psi_generator.py : phi -> psiの圧縮. JAXを利用

# Dependency
## Environment
### Simulation and Experiment
- ubuntu16.04
- ros kinetic
### Generate Psi and Simulation
- ubuntu20.04
- ros noetic

## Install
```
git clone https://github.com/htnk-lab/bebop_hatanaka_lab.git
git clone https://github.com/htnk-lab/persistent_coverage_control.git
# git clone https://github.com/htnk-lab/sky_camera 
```
- Follow the README of https://github.com/htnk-lab/bebop_hatanaka_lab.git

### angle_aware_switch
- [pytorch install](https://pytorch.org/get-started/locally/)
- YOLO v5
```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

# Build
```
catkin build
```

# Contents
- angle_aware_aruco : 土方さん農場での実験1. ArUcoマーカーとAngleAwareの実験
- angle_aware_avoid_tree : 木の回避をAngleAwareに追加
- angle_aware_camera : カメラ角度まで考慮するアルゴリズムの簡略版。論文のものとは少し違うので注意
- angle_aware_control : AngleAware
- angle_aware_debug : debug用
- angle_aware_switch : パトロール＋AngleAware
- angle_aware_unity : Unityとの連携
- bebop_aruco : ArUcoマーカーを用いた自己位置推定


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
If you can use JAX, then launch the following file. This is twice faster than generate_psi_no_jax.launch.
```
roslaunch angle_aware_control generate_psi.launch
```

## Make a file
Preparing
- Monitor
- Usb camera
- Smart phone
1. Launch (see [Experiment])
1. Make the rviz fullscreen
1. Launch the SimpleScreenRecorder and select the rviz window
1. Start recording
1. Run

