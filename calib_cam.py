# This file is used to generate the camera calibration file.

import camera

# Make a list of calibration images
file_name_pattern = '.\\camera_cal\\calibration*.jpg'

calib = camera.Calibrator()
calib.calibrate(file_name_pattern, 9, 6)
calib.save_calibration("cam_calib.p")
