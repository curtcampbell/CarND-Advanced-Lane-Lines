import camera
import process

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# Assume all images have the same shape.
#
def draw_lane(camera_calibarator, image, left_fit, right_fit):
    """
    
    :param camera_calibarator: Instance of Calibrator class used for distorting
           and undistoriting frame images.
    :param image: unwarped frame image. 
    :param left_fit: coefficients for curve fit of left lane in warped coordinates  
    :param right_fit: coefficients for curv fit of right lane in warped coordinates
    :return: Image with lane lines drawn on it.  
    """

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = camera_calibarator.unwarp(warp_zero)

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result


calib = camera.Calibrator()
calib.load_calibaration("cam_calib.p")

image = mpimg.imread(".\\test_images\\straight_lines2.jpg")

dst = calib.undistort(image)
gray_dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:,:,2]

# roi = np.array([[(0,433),
#                  (1279,433),
#                  (1279, 638),
#                  (0,638)]], dtype=np.int32)

# Choose a Sobel kernel size
ksize = 9 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
# gradx = process.abs_sobel_thresh(hsv, orient='x', sobel_kernel=ksize, thresh=(30, 100))
# grady = process.abs_sobel_thresh(hsv, orient='y', sobel_kernel=ksize, thresh=(30, 100))
# mag_binary = process.mag_thresh(hsv, sobel_kernel=ksize, mag_thresh=(30, 100))
# dir_binary = process.dir_threshold(hsv, sobel_kernel=ksize, thresh=(0.7, 1.3))


gradx = process.abs_sobel_thresh(gray_dst, orient='x', sobel_kernel=ksize, thresh=(30, 100))
grady = process.abs_sobel_thresh(gray_dst, orient='y', sobel_kernel=ksize, thresh=(30, 100))
mag_binary = process.mag_thresh(gray_dst, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = process.dir_threshold(gray_dst, sobel_kernel=ksize, thresh=(0.7, 1.3))
h_bin = process.hls_select(dst, thresh=(215, 255))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1


warped_image = calib.warp(combined)

left_fit, right_fit = process.fit_lanes(warped_image)

output = draw_lane(calib, image, left_fit, right_fit)

plt.imshow(output)
plt.show()

# window settings
window_width = 50
window_height = 80  # Break image into 9 vertical layers since image height is 720
margin = 100  # How much to slide left and right for searching


window_centroids = process.find_window_centroids(warped_image, window_width, window_height, margin)

# # Fit a second order polynomial to each
# # left_fit = np.polyfit(lefty, leftx, 2)
# # right_fit = np.polyfit(righty, rightx, 2)
#
# # If we found any window centers
# if len(window_centroids) > 0:
#
#     # Points used to draw all the left and right windows
#     l_points = np.zeros_like(warped_image)
#     r_points = np.zeros_like(warped_image)
#
#     # Go through each level and draw the windows
#     for level in range(0, len(window_centroids)):
#         # Window_mask is a function to draw window areas
#         l_mask = process.window_mask(window_width, window_height, warped_image, window_centroids[level][0], level)
#         r_mask = process.window_mask(window_width, window_height, warped_image, window_centroids[level][1], level)
#         # Add graphic points from window mask here to total pixels found
#         l_points[(l_points == 255) | ((l_mask == 1))] = 255
#         r_points[(r_points == 255) | ((r_mask == 1))] = 255
#
#     # Draw the results
#     template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
#     zero_channel = np.zeros_like(template)  # create a zero color channel
#     template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
#     warpage = np.array(cv2.merge((warped_image, warped_image, warped_image)),
#                        np.uint8)  # making the original road pixels 3 color channels
#     output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results
#
# # If no window centers found, just display orginal road image
# else:
#     output = np.array(cv2.merge((warped_image, warped_image, warped_image)), np.uint8)
#
# # Display the final results
# plt.imshow(output)
# plt.title('window fitting results')
# plt.show()

