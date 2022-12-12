# AngleAware2
This is the code for https://ieeexplore.ieee.org/document/9650560.

## Contents
**See the README in each package**
- angle_aware_aruco : 土方さん農場での実験1. ArUcoマーカーとAngleAwareの実験
- angle_aware_avoid_tree : 木の回避をAngleAwareに追加
- angle_aware_camera : カメラ角度まで考慮するアルゴリズムの簡略版。論文のものとは少し違うので注意
- **angle_aware_control : AngleAware base code**
- angle_aware_debug : debug用
- angle_aware_switch : パトロール＋AngleAware
- angle_aware_unity : Unityとの連携
- bebop_aruco : ArUcoマーカーを用いた自己位置推定


## Requirement
- ubuntu16.04 ros kinetic or
- ubuntu20.04 ros noetic


## Install
```
git clone https://github.com/htnk-lab/bebop_hatanaka_lab.git
```
- Follow the README of https://github.com/htnk-lab/bebop_hatanaka_lab.git

### Build
```
catkin build
```

## Usage
See each package.
angle_aware_control is the base package.



## Author

[Takumi Shimizu](https://github.com/tashiwater)




