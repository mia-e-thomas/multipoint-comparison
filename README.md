# **VIS-LWIR Feature Script**

## **Introduction**
To call the script, use the following line (append `-h` for help w/ parameters):

`python3 Optical_Thermal_Feature_Homography.py`

The purpose of this script is to compare point feature performance on VIS-LWIR image pairs from the multipoint dataset.

## **Description**
Three features tested: 
1. SIFT
2. ORB
3. MFD (Multispectral Feature Descriptor)

For *each* feature, the script:

*Matching*
* Instantiates feature
* Gets keypoints and descriptors from images
* Matches with the Brute Force Matcher (w/ `CrossCheck = True`)
* Times feature detection, description, and matching

*Drawing*
* Draws matches
* Saves image w/ matches

*Homography*
* Checks for sufficient number of matches (4 minimum for homography)
* Gets homography w/ mask of inliers and outliers
* Prints homography

*Re-Draw*
* Redraws matches with inliers in green and outliers in red
* Saves image

*Error Calculations*

NOTE: Since MultiPoint images aligned, locations of matching should be the same in soure and destination images. 
Error is calculated as Euclidean distance between src and dst points.
* Get vector of errors for each match
* Take average
* Get vector of *inlier* match errors
* Take average
* Get vector of *outlier* match errors

*Histogram*
* Histogram of ONLY inliers
* (Stacked) histogram of all points (w/ inliers green and outliers red)
* Save images

*Stats*
* Total matches
* Total inliers
* Total Outliers

*Saving*

Previously, I had used a .npz file to save the results. However, since revamping the code, I had to scrap that and have not implemented a new way to save the results.


## **Next Steps**

Future work on this will include the following:

*Refactor*
* Implement new way to save stats
* Move feature computation into loop (currently loop starts w/ homography step)
* Make the script a function that takes in a descriptor and its name
* **Run in larger batches** (currenly only one image pair at a time)

*Pre-processing*
* Histogram equalization
* Canny edge detector

*Additional Features*
* Include other hand-crafted, 'multispectral' features (EOH, LGHD)
* Include learned feature (MultiPoint, other learned feature)

*RANSAC Parameters*
* Try implementing "double" RANSAC where you apply RANSAC on the inliers

*Other*
* Apply homography before matching (will need to randomly sample homography space)
* Try to recover the inverse homography with matches (this section will require updating the error computation, but this should be straight-forward)


