**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./output_images/Calib-1-Distorted_Image.png "Original Distorted Image"
[image1]: ./output_images/Calib-2-Undistorted_Image.png "Undistorted"
[image11]: ./output_images/Pipeline-1-input.png "Road Transformed"
[image2]: ./output_images/Pipeline-2-Remove_Camera_distortion.png "Road Transformed"
[image3]: ./output_images/Pipeline-3-HSV_colorspace_thresholding.png "Binary Example"
[image4]: ./output_images/Pipeline-4-Perspective_transform.png "Warp Example"
[image5]: ./output_images/Pipeline-5-Curve_fit_overlay.png "Fit Visual"
[image6]: ./output_images/Pipeline-5-Curve_fit_overlay.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.   

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration code is found in `camera.py` A class called `CameraCalibrator` contains the implementation for
compensating for camera lens distortion.  This class wraps calls to the OpenCV library.  The following OpenCV functions 
are used:

* calibrateCamera()
* findChessboardCorners()
* undistort()

'calibrateCamera()' computes the matrix and distortion coefficients by comparing a set of coordinates from a distorted image
 with a set of known points representing actual measurements.

The following code snippet is found at line 35 in `campera.py` It loops through the images and stores the camera matrix 
and distortion coefficients.
```python
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

```
![alt text][image0]
![alt text][image1]

### Pipeline (single images)
Here I will describe various stages of the processing pipeline.  Each subsection will describe processing applied to 
input images in order to find the lane in the video stream.

We start with the input image below
![alt test][image11]

#### 1. Provide an example of a distortion-corrected image.
I read in the camera matrix and distortion coefficients and apply them to the image.
```python
class CameraCalibrator:
    ...   
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
```
```python
def get_top_down_mask(camera_calibrator, image):
    dst = camera_calibrator.undistort(image)
    ...
```
As you can see, the first call made in `get_top_down_mask()` is to undistort the image. The result is below.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I chose to simply convert images to HSV color space and then filter within a certain threshold. This seemed to 
work quite well.  
The code for this can be found in `process.py` 

```python

def hls_select(image, thresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    s_channel = hls[:,:,2]

    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    return binary_output
```

This code is called from with `get_top_down_mask()` line# 78 also in `process.py`

```python
def get_top_down_mask(camera_calibrator, image):
...
    # After all of the experimentation, it turns out for this particular project, hls performed the best.
    h_bin = hls_select(dst, thresh=(215, 255))
...

    return warped_image
```
The result is below.
![alt text][image3]
 
*Please Note When reading through `process.py` you will see commented out code as
well as functions for computing gradients that are not used.  Generally it's a good idea to removed unused code, but
in this case I wanted to leave it in to illustrate the methods I tried, as well as to facilitate future experimentation.*


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is found in `camp.py` around line 68.  I

```python
    def get_perspective_transform(self, is_inverse=False):

        src_pts = np.float32([[262, 684], [586, 457], [695, 457], [1019, 684]])
        dst_pts = np.float32([[262, 684], [262, 0], [1048, 0], [1048, 684]])
        if(is_inverse == False):
            return cv2.getPerspectiveTransform(src_pts, dst_pts)
        else:
            return  cv2.getPerspectiveTransform(dst_pts, src_pts)

    def warp(self, image):
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.perspective_mtx, img_size)

    def unwarp(self, image):
        img_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.inverse_perspective_mtx, img_size)

```

```python
def get_top_down_mask(camera_calibrator, image):
    dst = camera_calibrator.undistort(image)
...
    # After all of the experimentation, it turns out for this particular project, hls performed the best.
    h_bin = hls_select(dst, thresh=(215, 255))
...
    warped_image = camera_calibrator.warp(h_bin)

    return warped_image
```

The points used to compute the perspective transform were hard coded.  I made this 
choice because the perspective would tend to be consistent.  I therefore picked some points I thought would
work and tweaked them experimentally.  The points I chose are in the table below

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 262, 684      | 262, 684      | 
| 586, 457      | 262, 0        |
| 695, 457      | 1048, 0       |
| 1019, 684     | 1048, 684     |


The image blow shows the output of the transform
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find candidates for lanes I first computed a histogram across the X axis on the lower half of each picture frame.
I then determined the largest signal peaks as an indicator of where the lanes may be.  Using these peaks to determine 
the starting X position, I used a sliding window search to map out the lanes.  The search moves up the lane in the Y
direction centering each window about the centroid of the pixels clustered inside the window.  This search resulted in a
list of X,Y coordinates for each lane line.  A curve fit of these coordinates is done to produce quadratic polynomials 
for the left and right lane lines.

The code for this is in `fit_lanes()` in `process.py` around line# 203 

`fit_lanes()` is called from the `LaneDetector` class found in `detector.py` around line 50
```python
    def process_frame(self, image):

        top_down_image_mask = process.get_top_down_mask(self.camera_calibrator, image)

        ... Surrounding code here
                process.fit_lanes(top_down_image_mask, world_conversion_factor=self.world_conversion_factor)
```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This was done in the `calc_radius()` in `detector.py` line 166  Since the curve fit is done in pixels, a conversion is applied 
to transform the result into world units
```python
    def calc_radius(self, left_fit, right_fit):
        # y_eval = np.max(ploty)
        y_eval = 700 * self.world_conversion_factor[1]
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
        return left_curverad, right_curverad
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is implemented in `draw_lane()` within `detector.py`

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

To generate the camera calibration setting run calib_cam.py
To run the lane detection and create the output video, run lane_detection.py
To show the images displayed in this writeup, run writeup_artifacts.py

##### Additional things to note.

###### Threshold binary image
1. Although I chose to use a simple HSV color space threshold in this project, there are other ways to create the binary image mask. 
I chose HSV color space thresholding because it gave the best result on the test video.  I reason this is because 
the road lines tended to be rather bright in contrast to the road itself. This consistently produced a high value component for the
lane lines.  This was better than results I got from Sobel edge detection.  I believe this worked well in this situation but my degrade 
in other circumstances. If given more time, I would like to incorporate some filtering using the hue value as well.  
I think lane lines may tend to be within a certain hue range.

2. The approach still struggled in areas where harsh shadows were cast onto the road and where there were 
random markings on the road.  A black asphalt road wold work better than a concrete one.  Concrete is more susceptible to 
being marked up by various things like rubber skid marks.

###### Smoothing
1. I tried to smooth the presentation a bit taking a rolling average of the polynomial coefficients.  This worked for 
the most part, but it also has a tendency to lag the road when the curvature changes abruptly.  Also while the lane 
polygon is smoothed from frame to frame, the calculation for radius values are not.
 
###### Radius of Curvature
1. Areas where the highway was straight yeilded a large radius.  In my coded I considered an extremely large radius
to mean no curve at all.  On areas where the road was straight, my output stops showing a radius and displays 'No Curve'


