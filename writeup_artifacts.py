from camera import CameraCalibrator
import process
from detector import LaneDetector
import matplotlib.pyplot as plt


def plot(image, title=None, cmap=None):
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.imshow(image, cmap=cmap)


calib = CameraCalibrator()
calib.load_calibaration("cam_calib.p")
lane_detector = LaneDetector(calib)


# see calib_cam.py for the code that generates and saves the cam_calib.p
# cam_calib.p is pickeled data containing the tranform matrices use to
# undistort the images

distorted_image = plt.imread(".\\camera_cal\\calibration1.jpg")

undistorted_image =  calib.undistort(distorted_image)

plot(distorted_image, "Distorted calibration image")
plot(undistorted_image, "Undistorted calibration image")
plt.show()

# pipeline
image = plt.imread(".\\test_images\\test3.jpg")
test_image = plt.imread(".\\test_images\\test3.jpg")

correct_distortion_image = calib.undistort(test_image)
hls_threshold = process.hls_select(correct_distortion_image, thresh=(215, 255))
top_down = calib.warp(hls_threshold)

final_output = lane_detector.process_frame(test_image)

plot(image, "Pipeline-input")
plot(correct_distortion_image, "Pipeline-remove-camera-distortion")
plot(hls_threshold, "HLV color space threshold", cmap='gray')
plot(top_down, "Top down perspective transform", cmap='gray')
plot(final_output, "Comple curve fit")

plt.show()

