#!/usr/bin/env python

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

        self.sift = cv2.xfeatures2d.SIFT_create()

        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)


    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)

        #Load template image from file
        img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE) 
        self.kp_template, self.desc_template = self.sift.detectAndCompute(img, None)
        print("Loaded template image file: " + self.template_path)
        

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                     bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        # Takes a live image from the camera and saves it in frame
        ret, frame = self._camera_device.read()
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         # Detect features and descriptors of the webcam frame (query image)
        kp_query, desc_query = self.sift.detectAndCompute(grayframe, None)

        # Feature matching

       
        # Compare template with query image            
        matches = self.flann.knnMatch(desc_query, self.desc_template, k=2)
        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)

        ## Homography

        # Obtain matrix and implement findHomography
        query_pts = np.float32([kp_query[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        template_pts = np.float32([self.kp_template[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        
        if len(good_points) > 5:
            matrix, mask = cv2.findHomography(query_pts, template_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()

            if (matrix is not None):
                # Perspective transform
                h, w = grayframe.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                print(pts)
                print(matrix)

                dst = cv2.perspectiveTransform(pts, matrix)

                homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            else:
                homography = frame
        else:
            homography = frame

        pixmap = self.convert_cv_to_pixmap(homography)
        self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())

