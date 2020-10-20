import numpy as np
import cv2

# load the calibration matrix and distortion coefficients computed in camera_calibration.ipynb
mtx = np.load("mtx.npy")
dist_parameters = np.load("dist_parameters.npy")


def undistort_image(image):
    """"
    Undistort image with the read mtx and dist_parameters
    """
    undst = cv2.undistort(image, mtx, dist_parameters, None, mtx)

    return undst


def threshold(image, s_thresh=(150, 255), sx_thresh=(23, 150), xReduction=.05, yReduction=.58):
    """
    Filter image to obtein a bitmap with the way lanes
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Apply sobel x to l chanel
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &
             (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # combining thresholds
    bitmap = np.logical_or(s_binary, sxbinary)

    return bitmap
