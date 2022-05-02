import numpy as np
import cv2 
import os                                
import shutil                            
import h5py                              
import yaml
import argparse
import matplotlib.pyplot as plt

def main(): 

    #------------#
    # Parameters #
    #------------#
    # Define script arguments
    parser = argparse.ArgumentParser(description='Temp: show image pair')
    parser.add_argument('-y', '--yaml-config', default='config/config_feature_homography.yaml', help='YAML config file')
    parser.add_argument('-i', '--index', default=500, type=int, help='Index of image pair')
    parser.add_argument('-s', dest='show', action='store_true', help='If set, the prediction the images/plots are displayed')
    parser.add_argument('-n', '--num-matches', default=0, type=int, help='Shows the best \'n\' matches (default all)')

    # Get arguments
    args = parser.parse_args()

    # Get YAML file
    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Configure paths
    dataset_path = os.getcwd() + '/' + config['dir_dataset']
    results_path = os.getcwd() + '/' + config['dir_results']

    #----------#
    # Get Data # 
    #----------#

    #------- Get image pair from dataset -------#
    # Open hdf5 file
    f = h5py.File(dataset_path, 'r')

    # Get the keys (each key is a different timestep with all 3 images)
    timestamp_keys = list(f.keys())

    # Get the image types 
    im_types = list(f[timestamp_keys[0]].keys())

    # Select the timestamp and get the image trio
    timestamp = timestamp_keys[args.index]
    im_trio   = f[timestamp]

    # ------ Make folder for specific image pair ------ #
    folder_name = results_path + '/' + timestamp + '/'

    # Overwrite existing folder
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

    #---------#
    # Process # 
    #---------#

    #------- Stack thermal & optical, Save, Show -------#

    # Get RGB image and convert to uint8
    optical_im = im_trio['optical'][:][:]
    rgb_optical_original = cv2.cvtColor((optical_im*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Get thermal image and convert to uint8
    thermal_im = im_trio['thermal'][:][:]
    rgb_thermal_original = cv2.cvtColor((thermal_im*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Stack
    opt_therm_stack_original = np.hstack((rgb_optical_original,rgb_thermal_original))

    # Show
    if args.show:
        cv2.imshow('Optical (L) & Thermal (R) - ' + str(timestamp), opt_therm_stack_original)
        cv2.waitKey()
    
    # Save
    cv2.imwrite(folder_name+"Opt_Therm_Stacked-"+str(timestamp) +".png", opt_therm_stack_original)

    #------#
    # SIFT # 
    #------#

    #------- SIFT -------#
    # Instantiate SIFT detector
    # Default Params:
    # nfeatures = 0
    # nOctaveLayers = 3
    # contrastThreshold = 0.04
    # edgeThreshold = 10
    # sigma = 1.6
    # sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10) # DEFAULT PARAMS
    sift = cv2.SIFT_create(contrastThreshold=0.12, edgeThreshold=20) # MODIFIED PARAMS

    # Find SIFT keypoints and descriptors
    kp1_sift, des1_sift = sift.detectAndCompute(rgb_optical_original,None)
    kp2_sift, des2_sift = sift.detectAndCompute(rgb_thermal_original,None)

    # Instantiate Brute Force Matcher with default parameters
    # Default Params:
    # normType = NORM_L2 (use NORM_L2 for sift/surf & NORM_HAMMING for orb, brief, brisk)
    # crossCheck = False <--- If true, only returns "mutual best match" (i best for j AND j best for i)
    bf_sift = cv2.BFMatcher(crossCheck = True)

    # Find matches with Brute Force
    matches_sift = bf_sift.match(des1_sift,des2_sift)

    # Sort matches in order of distance
    matches_sift = sorted(matches_sift, key = lambda x:x.distance)

    # Take only the first 'n' matches (default all when param is zero)
    if args.num_matches: matches_sift = matches_sift[:args.num_matches]

    # Draw matches
    opt_therm_match_sift = cv2.drawMatches(
                                rgb_optical_original,kp1_sift,
                                rgb_thermal_original,kp2_sift,
                                matches_sift,None,
                                # flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                )

    # Show images
    cv2.imshow('Opt-Therm Match SIFT - ' + str(timestamp), opt_therm_match_sift)
    cv2.waitKey()

    # Save Image
    cv2.imwrite(folder_name+"Opt_Therm_Match_SIFT-"+str(timestamp) +".png", opt_therm_match_sift)

    #------- HOMOGRAPHY -------#
    src_pts_sift = np.float32([ kp1_sift[m.queryIdx].pt for m in matches_sift ]).reshape(-1,1,2)
    dst_pts_sift = np.float32([ kp2_sift[m.trainIdx].pt for m in matches_sift ]).reshape(-1,1,2)
    M_sift, mask_sift = cv2.findHomography(src_pts_sift, dst_pts_sift, cv2.RANSAC) # MODIFIED
    print("SIFT Homography: \n" + str(M_sift))
    

    #------- REDRAW INLIERS -------#
    # 1. Draw all matches in red
    opt_therm_match_sift = cv2.drawMatches(rgb_optical_original,kp1_sift,rgb_thermal_original,kp2_sift,matches_sift, None,
                                matchColor = (0,0,255),       # draw matches in red color
                                )
    # 2. Draw inliers in green 
    w = int(opt_therm_match_sift.shape[1]/2)
    opt_therm_match_sift = cv2.drawMatches(opt_therm_match_sift[:,:w,:],kp1_sift,opt_therm_match_sift[:,w:,:],kp2_sift,matches_sift, None,
                                matchColor = (0,255,0),       # draw matches in green color
                                matchesMask = mask_sift[:,0],  # draw only inliers
                                )

    # Show images
    cv2.imshow('SIFT Inliers and Outliers - ' + str(timestamp), opt_therm_match_sift)
    cv2.waitKey()

    # Save Image
    cv2.imwrite(folder_name+"Opt_Therm_Match_SIFT_Inliers_Outliers-"+str(timestamp) +".png", opt_therm_match_sift)

    #-----#
    # ORB # 
    #-----#

    #------- ORB -------#
    # Instantiate ORB detector
    # Default Params:
    # nfeatures = 500
    # scaleFactor = 1.2f
    # nlevels = 8
    # edgeThreshold = 31
    # firstLevel = 0
    # WTA_K = 2
    # scoreType = ORB::HARRIS_SCORE
    # patchSize = 31
    # fastThreshold = 20
    # orb = cv2.ORB_create(edgeThreshold = 31, patchSize = 31, fastThreshold= 20) # DEFAULT PARAMETERS 
    orb = cv2.ORB_create(edgeThreshold = 91, patchSize = 60, fastThreshold= 20) # MODIFIED PARAMETERS 

    # Find SIFT keypoints and descriptors
    kp1_orb, des1_orb = orb.detectAndCompute(rgb_optical_original,None)
    kp2_orb, des2_orb = orb.detectAndCompute(rgb_thermal_original,None)

    # Instantiate Brute Force Matcher with default parameters
    # Default Params:
    # normType = NORM_L2 (use NORM_L2 for sift/surf & NORM_HAMMING for orb, brief, brisk)
    # crossCheck = False <--- If true, only returns "mutual best match" (i best for j AND j best for i)
    bf_orb = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)

    # Find matches with Brute Force
    matches_orb = bf_orb.match(des1_orb,des2_orb)

    # Sort matches in order of distance
    matches_orb = sorted(matches_orb, key = lambda x:x.distance)

    # Take only the first 'n' matches (default all when param is zero)
    if args.num_matches: matches_orb = matches_orb[:args.num_matches]

    # Draw matches
    opt_therm_match_orb = cv2.drawMatches(
                                rgb_optical_original,kp1_orb,
                                rgb_thermal_original,kp2_orb,
                                matches_orb,None,
                                # flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                )

    # Show images
    cv2.imshow('Opt-Therm Match ORB - ' + str(timestamp), opt_therm_match_orb)
    cv2.waitKey()

    # Save Image
    cv2.imwrite(folder_name+"Opt_Therm_Match_ORB-"+str(timestamp) +".png", opt_therm_match_orb)

    #------- HOMOGRAPHY -------#
    src_pts_orb = np.float32([ kp1_orb[m.queryIdx].pt for m in matches_orb ]).reshape(-1,1,2)
    dst_pts_orb = np.float32([ kp2_orb[m.trainIdx].pt for m in matches_orb ]).reshape(-1,1,2)
    M_orb, mask_orb = cv2.findHomography(src_pts_orb, dst_pts_orb, cv2.RANSAC)
    print("ORB Homography: \n" + str(M_orb))

    #------- REDRAW INLIERS -------#
    # 1. Draw all matches in red
    opt_therm_match_orb = cv2.drawMatches(rgb_optical_original,kp1_orb,rgb_thermal_original,kp2_orb, matches_orb, None,
                                matchColor = (0,0,255), # draw matches in red color
                                )
    # 2. Draw inliers in green 
    w = int(opt_therm_match_orb.shape[1]/2)
    opt_therm_match_orb = cv2.drawMatches(opt_therm_match_orb[:,:w,:],kp1_orb,opt_therm_match_orb[:,w:,:],kp2_orb,matches_orb, None,
                                matchColor = (0,255,0),       # draw matches in green color
                                matchesMask = mask_orb[:,0],  # draw only inliers
                                )

    # Show images
    cv2.imshow('ORB Inliers and Outliers - ' + str(timestamp), opt_therm_match_orb)
    cv2.waitKey()

    # Save Image
    cv2.imwrite(folder_name+"Opt_Therm_Match_ORB_Inliers_Outliers-"+str(timestamp) +".png", opt_therm_match_orb)



if __name__ == "__main__":
    main()