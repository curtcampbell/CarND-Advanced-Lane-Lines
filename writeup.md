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

The camera calibartion code is found in `camera.py` A class called `CameraCalibrator` contains the implementation for
compensating for camera lens distortion.  This class wraps calls to the OpenCV library.  The following OpenCV functions 
are used:

* calibrateCamera
* findChessboardCorners
* undistort

'calibrateCamera' computes the matrix and distortion coefficients by comparing a set of coordinates from a distorted image
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
This section will describe various stages of the processing pipeline.  Each section will describe processing applied to 
input images in order to find the lane in the video stream.

The image below is the original image as present by the camera.  Subsequent images will depect the various transforms that
occur  during the lane finding process.

![alt test][image11]

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Although there are a few methods for creating the binary image mask including computing different kinds of gradients.  
I found however I could simply convert images to HSV color space and then filter within a certain threshold. This seemed to 
work quite well.  This is because in the project sample, the road lines tended to be rather bright which consistently produced
a high value component.  It's understood that this worked well in this situation but my degrade in other circumstances.

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

The result is below.
![alt text][image3]
 

*Please Note When reading through `process.py` you will see commented out code as
well as functions for computing gradients that are not used.  Generally it's a good idea to removed unused code, but
in this case I wanted to leave it in to illustrate the methods I tried, as well as to facilitate future experimentation.*



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is also found in `camp.py` around line 68.  I

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

The points used to compute the perspective transform were hard coded.  I made this 
choice because I determined that the perspective would always be consistent.  I therefore picked some points I thought would
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

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
