import numpy as np
import cv2

# load the calibration matrix and distortion coefficients computed in camera_calibration.ipynb
mtx_dist = np.load("mtx_dist.npy")
dist_parameters = np.load("parameters_dist.npy")
mtx_perspective = np.load("mtx_perspective.npy")
mtx_inv_perspective = np.load("mtx_inv_perspective.npy")

# approximated rattio of real distance length agains pixels of the eagle eye images
ym_per_pix = 3 / (590 - 555)
xm_per_pix = 3.7 / (1040 - 270)


def undistort_image(image):
    """"
    Undistort image with the read mtx and dist_parameters
    """
    undst = cv2.undistort(image, mtx_dist, dist_parameters, None, mtx_dist)

    return undst


def threshold(image, s_thresh=(150, 255), sx_thresh=(23, 150)):
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
    bitmap = np.logical_or(s_binary, sxbinary).astype(np.uint8)

    return bitmap


def eagle_eye(image):
    """
    Proyect the image to the eagle eye view

    Note: it is asumed that the image was previously undistorted
    """
    img_size = (image.shape[1], image.shape[0])

    return cv2.warpPerspective(image, mtx_perspective, img_size, flags=cv2.INTER_LINEAR)


def eagle_eye_inv(image):
    """
    Proyect the eagle eye to the normal view
    """
    img_size = (image.shape[1], image.shape[0])

    return cv2.warpPerspective(image, mtx_inv_perspective, img_size, flags=cv2.INTER_LINEAR)


def find_lane_pixels_window(binary_warped, nwindows=10, margin=100, minpix=40):
    """
    Finds the pixels which are part of a lane using a moving window metod. The binary warp
    represents a binary map of the lanes in eagle view
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        # Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window ###
        if minpix < len(good_left_inds):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if minpix < len(good_right_inds):
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fit, right_fit, left_fitx, right_fitx, ploty


def fit_around_poly(binary_warped, left_fit, right_fit, margin=100):
    """
    This function fits a polynomial using the previous computed fitting as parameter.
    """
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the area of search which is (x - x_fit)^2 < margin^2
    left_lane_inds = (
        (nonzerox - (left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2]))**2 < margin**2).nonzero()[0]
    right_lane_inds = (
        (nonzerox - (right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2]))**2 < margin**2).nonzero()[0]

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(
        binary_warped.shape, leftx, lefty, rightx, righty)

    return left_fit, right_fit, left_fitx, right_fitx, ploty


def fit_to_real_space(left_fit, right_fit, ym_per_pix=ym_per_pix, xm_per_pix=xm_per_pix):
    '''
    tranforms the fit from pixel space to real world space
    '''
    left_fit_real = np.zeros_like(left_fit)
    right_fit_real = np.zeros_like(right_fit)

    left_fit_real[0] = xm_per_pix * left_fit[0] / ym_per_pix**2
    left_fit_real[1] = xm_per_pix * left_fit[1] / ym_per_pix
    left_fit_real[2] = xm_per_pix * left_fit[2]

    right_fit_real[0] = xm_per_pix * right_fit[0] / ym_per_pix**2
    right_fit_real[1] = xm_per_pix * right_fit[1] / ym_per_pix
    right_fit_real[2] = xm_per_pix * right_fit[2]

    return left_fit_real, right_fit_real


def measure_curvature(ploty, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in the given dimensions.
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    A_left = left_fit[0]
    B_left = left_fit[1]
    A_right = right_fit[0]
    B_right = right_fit[1]

    # Calculation of R_curve (radius of curvature)
    left_curverad = np.power(
        1 + np.power(2*A_left*y_eval + B_left, 2), 1.5) / (2 * np.abs(A_left))
    right_curverad = np.power(
        1 + np.power(2*A_right*y_eval + B_right, 2), 1.5) / (2 * np.abs(A_right))

    return left_curverad, right_curverad


def car_position(img, ploty_real, left_fit_real, right_fit_real, offset=1.398877002384628):
    """
    Return car position relative to the center of the lane
    """
    y_bottom = np.max(ploty_real)
    left_lane_x_position = left_fit_real[0] * y_bottom**2 + \
        left_fit_real[1] * y_bottom + left_fit_real[2]
    right_lane_x_position = right_fit_real[0] * y_bottom**2 + \
        right_fit_real[1] * y_bottom + right_fit_real[2]
    middle_position = xm_per_pix * img.shape[0] / 2

    possition = middle_position - \
        (right_lane_x_position + left_lane_x_position) / 2 + offset

    return possition
