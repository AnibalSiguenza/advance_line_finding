import numpy as np
import cv2

# load the calibration matrix and distortion coefficients computed in camera_calibration.ipynb
mtx = np.load("mtx.npy")
dist_parameters = np.load("dist_parameters.npy")


def undistort_image(img):
    undst = cv2.undistort(img, mtx, dist_parameters, None, mtx)

    return undst
