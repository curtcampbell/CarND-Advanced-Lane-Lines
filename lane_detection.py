from camera import CameraCalibrator
from detector import LaneDetector

calib = CameraCalibrator()
calib.load_calibaration("cam_calib.p")
lane_detector = LaneDetector(calib)

lane_detector.detect_lanes('.\\project_video.mp4', '.\\project_video_output.mp4')
#lane_detector.detect_lanes('.\\challenge_video.mp4', '.\\challenge_video_output.mp4')


