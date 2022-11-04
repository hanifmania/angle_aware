# bebop_aruco
AR markerを用いた自己位置推定

# Contents
- src
    - ar_detect.py : cv2 lapper
    - camera_from_ar.py : カメラ位置推定
    - camera2drone.py : カメラ位置からdrone位置への変換
# Dependency
## Environment
- ubuntu16.04
- ros kinetic

## Requirements
```
git clone https://github.com/htnk-lab/persistent_coverage_control.git
pip3 install img2pdf
```

# Usage
## PC camera debug
```
roslaunch bebop_aruco debug_pc_bringup.launch
```

# Knowledge
## bebop camera
- bebop2はジンバルを搭載し、地面との角度が常に指令値となるよう自動調整している。
- camera_controlの定義域は [-82, 5?] deg