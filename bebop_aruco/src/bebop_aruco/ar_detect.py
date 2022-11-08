#!usr/bin/env python
# -*- coding: utf-8 -*-
from cv2 import aruco
import cv2
import numpy as np

class ARDetect():
    def str2ar_type(self, name_str):
        return getattr(aruco, name_str)
    
    def get_dictionary(self, ar_dictionary):
        if type(ar_dictionary) is str:
            # 文字列で指定されたら数値に変換する
            ar_dictionary = self.str2ar_type(ar_dictionary)
        return aruco.getPredefinedDictionary(ar_dictionary)

    def set_dictionary(self, ar_dictionary):
        """_summary_

        Args:
            ar_dictionary (str or int): ARの種類(https://docs.opencv.org/3.4/d9/d6a/group__aruco.html#gac84398a9ed9dd01306592dd616c2c975)
        """        
        self._dictionary = self.get_dictionary(ar_dictionary)

    def set_params(self, marker_length, ar_dictionary, _camera_matrix, distort_coeff):
        """_summary_

        Args:
            marker_length (float): 黒枠を含む一辺の大きさ[m]
            ar_dictionary (str or int): ARの種類(https://docs.opencv.org/3.4/d9/d6a/group__aruco.html#gac84398a9ed9dd01306592dd616c2c975)
            _camera_matrix (ndarray): カメラ行列
            distort_coeff (ndarray): 歪行列
        """        
        self.set_dictionary(ar_dictionary)
        self._marker_length = marker_length
        self._camera_matrix = _camera_matrix
        self._distort_coeff = distort_coeff

    def set_borad(self, markersX, markersY, markerSeparation):
        """cv.aruco.GridBoard_create

        Args:
            markersX (int): number of markers in X direction
            markersY (int): number of markers in Y direction
            markerSeparation (float): separation between two markers (same unit as markerLength)
        """        
        self._board = aruco.GridBoard_create(markersX, markersY, self._marker_length, markerSeparation, self._dictionary)

    def find_marker(self):
        self.corners, self.ids, self.rejectedImgPoints = aruco.detectMarkers(self.img,self._dictionary)
        return self.ids
        
    def get_ar_detect_img(self):
        im_copy = self.img.copy()
        aruco.drawDetectedMarkers(im_copy, self.corners, self.ids, (0,255,0))
        return im_copy

    def show(self):
        im = self.get_ar_detect_img()
        cv2.imshow("result", im)
        
    def get_ar_posi(self, id):
        corner = self.get_corner(id)
        return self.get_posi(corner)

    def get_posi(self, corner):
        if len(corner)  < 1:
            return [], []
        ret = aruco.estimatePoseSingleMarkers([np.array(corner, dtype = float)],self._marker_length, self._camera_matrix, self._distort_coeff)
        rvec, tvec = ret[0], ret[1]
        return rvec[0][0], tvec[0][0]

    def marker_num(self):
        return len(self.corners)
    
    def get_id_index(self,id):
        index = np.where(self.ids == id)[0]
        if len(index) < 1:
            return None
        return index[0]

    def get_corner(self, id):
        index = self.get_id_index(id)
        if index == None:
            return []
        ret = self.corners[index][0]
        return ret
    
    def get_camera_pose(self, id):
        """ARからcameraの座標を推定する

        Args:
            id (int): AR id

        Returns:
            ndarray: xyz
            ndarray: rpy(正面から撮影しているときに[0,0,0])
        """       
        corner = self.get_corner(id)
        # rospy.loginfo(corner)
        if len(corner)  < 1:
            return [], []
        ret = aruco.estimatePoseSingleMarkers([np.array(corner, dtype = float)],self._marker_length, self._camera_matrix, self._distort_coeff)
        # cameraから見たARのpose. AR2camera  
        rvec, tvec = ret[0], ret[1]
        # print("rvec")
        # print(rvec)
        # print(tvec)
        ### R : AR2camera
        R = cv2.Rodrigues(rvec)[0]  # 回転ベクトル -> 回転行列

        # print("R")
        # print(R)
        # return tvec.reshape(-1), R
        R_T = R.T #camera2AR. ARからみたcamera
        T = tvec[0].T 
        # rospy.loginfo(-T)
        # ARからみたcamera座標系の原点
        xyz = np.dot(R_T, - T).squeeze()
        # rpy = np.deg2rad(cv2.RQDecomp3x3(R_T)[0])
        return xyz, R_T
    
    def set_img(self, img):
        self.img = img
    

    def estimatePoseBoard(self):
        """_summary_

        Returns:
            _type_: _description_
        """        # cameraから見たARのpose. AR2camera
        retval, rvec, tvec = aruco.estimatePoseBoard(self.corners, self.ids, self._board, self._camera_matrix, self._distort_coeff)

        rvec_np = np.array(rvec).reshape(-1)
        R, Jacob = cv2.Rodrigues(rvec_np)  # 回転ベクトル -> 回転行列
        ### R : AR2camera
        print("rvec, tvec")
        print(rvec)
        print(tvec)
        # print("R")
        # print(R)
        R_T = R.T #camera2AR
        T = tvec #camera 座標系からみたARの原点
        xyz = np.dot(R_T, - T).squeeze()
        rpy = np.deg2rad(cv2.RQDecomp3x3(R_T)[0])
        return xyz, rpy

if __name__ == "__main__":
    ar_marker_size = 0.03 #ARマーカー一辺. 黒い外枠含む[m]
    _camera_matrix = np.loadtxt("../../data/camera/cameraMatrix.csv", delimiter= ",")
    distCoeffs = np.loadtxt("../../data/camera/distCoeffs.csv", delimiter= ",")
    ar = ARDetect()
    ar.set_params(ar_marker_size, aruco.DICT_4X4_50, _camera_matrix, distCoeffs)
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    print("capture")
    while True:
        _, frame = cap.read()
        ar.set_img(frame)
        #ARマーカー検出
        ar.find_marker()
        ar.show()
        # ar.get
        # rvec, posis = ar.get_ar_posi(0)
        # print(posis)

        xyz, rpy = ar.get_camera_pose(0)
        print(rpy)
        cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()

