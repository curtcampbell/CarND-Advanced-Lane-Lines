import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob, pickle

from skimage.transform._geometric import warp


def find_corners(img, nx, ny):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    return cv2.findChessboardCorners(gray, (nx, ny), None)


class CameraCalibrator:

    mtx = None
    dist = None
    rvecs = None
    tvecs = None
    perspective_mtx = None

    def __init__(self):
        self.img_points = []
        self.obj_points = []
        self.perspective_mtx = self.get_perspective_transform()

    def save_calibration(self, file_name):
        pickle.dump(self, open(file_name, "wb"))

    def load_calibaration(self, file_name):
        calib = pickle.load( open( file_name, "rb" ) )
        self.mtx = calib.mtx
        self.dist = calib.dist
        self.rvecs = calib.rvecs
        self.tvecs = calib.tvecs

    def calibrate(self, image_name_pattern, nx, ny):
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        images = glob.glob(image_name_pattern)

        for image in images:

            img = mpimg.imread(image)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            if ret == True:
                self.img_points.append(corners)
                self.obj_points.append(objp)

        ret, self.mtx, self.dist, self.rvecs, self.tvecs =\
            cv2.calibrateCamera(self.obj_points, self.img_points, gray.shape[::-1], None, None)

    def get_perspective_transform(self):

        src_pts = np.float32([[559, 472], [725, 472], [265, 667], [1040, 668]])
        dst_pts = np.float32([[265, 472], [1040, 472], [265, 667], [1040, 668]])
        return cv2.getPerspectiveTransform(src_pts, dst_pts)

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def warp(self, image):
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.perspective_mtx, img_size)



def abs_sobel_thresh(gray_img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if(orient == 'x'):
        sobel= cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, sobel_kernel)

    # Apply threshold
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def mag_thresh(gray_image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx= cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelxy = np.absolute(sobelx + sobely)
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))

    # Apply threshold
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return mag_binary


def dir_threshold(gray_image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx= cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    absdir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(absdir)
    dir_binary[(absdir >= thresh[0]) & (absdir <= thresh[1])] = 1

    # Apply threshold
    return dir_binary


def hls_select(image, thresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


# Make a list of calibration images
# fname_pattern = '.\\camera_cal\\calibration*.jpg'

# calib = CameraCalibrator()
# calib.calibrate(fname_pattern, 9, 6)
# calib.save_calibration("cam_calib.p")

calib2 = CameraCalibrator()
calib2.load_calibaration("cam_calib.p")

image = mpimg.imread(".\\test_images\\test5.jpg")

dst = calib2.undistort(image)
gray_dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)

# Choose a Sobel kernel size
ksize = 9 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(gray_dst, orient='x', sobel_kernel=ksize, thresh=(30, 100))
grady = abs_sobel_thresh(gray_dst, orient='y', sobel_kernel=ksize, thresh=(30, 100))
mag_binary = mag_thresh(gray_dst, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(gray_dst, sobel_kernel=ksize, thresh=(0.7, 1.3))
h_bin = hls_select(dst, thresh=(140, 255))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

warped_image = calib2.warp(h_bin)
plt.figure()
plt.imshow(image)

plt.figure()
plt.imshow(warped_image, cmap='gray')

plt.figure()
plt.imshow(combined, cmap='gray')

plt.figure()
plt.imshow(h_bin, cmap='gray')

plt.show()


# sobel(dst, 25, 150, 21)

# sobel_direction_mask(dst, 0.7, 1.3, 3)
# plt.figure()
# plt.imshow(image)
# plt.figure()
# plt.imshow(dst)
# plt.show()


