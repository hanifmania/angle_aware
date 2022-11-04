#!usr/bin/env python
# -*- coding: utf-8 -*-

### calibration用プログラム。カメラ行列等をcsvに出力する

import numpy as np
import cv2
# from glob import glob
import Tkinter
import tkMessageBox
import rospy

square_size = 18     # 正方形のサイズ
pattern_size = (11-1, 8-1)  # 模様のサイズ
pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 ) #チェスボード（X,Y,Z）座標の指定 (Z=0)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
obj_points = []
img_points = []
print(pattern_points)
cap = cv2.VideoCapture(0) #ビデオキャプチャの開始
cap.set(3, 1280)
cap.set(4, 720)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# for i in glob("*.jpg"):
for i in range(100):
    if rospy.is_shutdown():
        break
    fn = str(i)
    # 画像の取得
    # im = cv2.imread(fn, 0)
    for _ in range(5):
        _, frame = cap.read()
    # _, frame = cap.read()
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print("loading..." + fn)
    # チェスボードのコーナーを検出
    found, corner = cv2.findChessboardCorners(im, pattern_size)
    cv2.imshow("img",im)
    # コーナーがあれば
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(im, corner, (5,5), (-1,-1), term)    #よくわからないがサブピクセル処理（小数点以下のピクセル単位まで精度を求める）
        cv2.drawChessboardCorners(im, pattern_size, corner,found)
        cv2.imshow('found corners in' + fn,im)
    # コーナーがない場合のエラー処理
    cv2.waitKey(100)
    if not found:
        print('chessboard not found')
        continue
    # 選択ボタンを表示
    root = Tkinter.Tk()
    root.withdraw()
    if tkMessageBox.askyesno('askyesno','この画像の値を採用しますか？'):
        img_points.append(corner.reshape(-1, 2))   #appendメソッド：リストの最後に因数のオブジェクトを追加 #corner.reshape(-1, 2) : 検出したコーナーの画像内座標値(x, y)
        obj_points.append(pattern_points)
        print('found corners in ' + fn + ' is adopted')
    else:
        print('found corners in' + fn + ' is not adopted')
    cv2.destroyAllWindows()
    if len(img_points) > 0:
        break
cap.release()
    
# 内部パラメータを計算
focus_length = 1090.0 #[mm]
sensor_width = 1410.0 #[mm]
f_x = im.shape[1] * focus_length/ sensor_width
f_y = f_x
c_x = im.shape[1] /2.0
c_y = im.shape[0] /2.0
init_camera_matrix = np.array([[f_x, 0, c_x],[0, f_y, c_y],[0, 0, 1]])
allflags = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO
print(init_camera_matrix)
print(allflags)
try:
    rms, cameraMatrix, distCoeffs, r, t = cv2.calibrateCamera(obj_points,img_points,(im.shape[1],im.shape[0]),init_camera_matrix, None, flags=allflags)
except Exception as e:
    print(e)
# 計算結果を表示
print("RMS = ", rms)
print("K = \n", cameraMatrix)
print("d = ", distCoeffs.ravel())
# 計算結果を保存
np.savetxt("cameraMatrix.csv", cameraMatrix, delimiter =',',fmt="%0.14f") #カメラ行列の保存
np.savetxt("distCoeffs.csv",distCoeffs, delimiter =',',fmt="%0.14f") #歪み係数の保存
