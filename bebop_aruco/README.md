# bebop_aruco
AR markerを用いた自己位置推定

## Demo
![gif](https://user-images.githubusercontent.com/49784413/205934399-fef364bc-57ce-4aa3-a017-b2a13ddc7f4c.gif)

## Install
```
pip3 install img2pdf
```

## Run test
### PC camera debug
```
roslaunch bebop_aruco debug_pc_bringup.launch
```

## Knowledge
### bebop camera
- bebop2はジンバルを搭載し、地面との角度が常に指令値となるよう自動調整している。
- camera_controlの定義域は [-82, 5?] deg


## Contents
- scripts:
    - ar_map_generator.py : AR marker印刷用pdfを生成
    - calibration.py : カメラキャリブレーション
- src
    - ar_detect.py : cv2 lapper
    - ar2camera.py : カメラ位置推定
    - camera2drone.py : カメラ位置からdrone位置への変換

## Author

[Takumi Shimizu](https://github.com/tashiwater)

