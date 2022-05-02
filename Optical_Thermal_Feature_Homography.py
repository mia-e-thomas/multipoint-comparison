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
    kp1, des1 = sift.detectAndCompute(rgb_optical_original,None)
    kp2, des2 = sift.detectAndCompute(rgb_thermal_original,None)

    # Instantiate Brute Force Matcher with default parameters
    # Default Params:
    # normType = NORM_L2 (use NORM_L2 for sift/surf & NORM_HAMMING for orb, brief, brisk)
    # crossCheck = False <--- If true, only returns "mutual best match" (i best for j AND j best for i)
    bf = cv2.BFMatcher(crossCheck = True)

    # Find matches with Brute Force
    matches = bf.match(des1,des2)

    # Sort matches in order of distance
    matches = sorted(matches, key = lambda x:x.distance)

    # Take only the first 'n' matches (default all when param is zero)
    if args.num_matches: matches = matches[:args.num_matches]

    # Draw matches
    opt_therm_sift_match = cv2.drawMatches(rgb_optical_original,kp1,
                                rgb_thermal_original,kp2,
                                matches,None,
                                # flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                )

    # Show images
    cv2.imshow('Opt-Therm Match SIFT - ' + str(timestamp), opt_therm_sift_match)
    cv2.waitKey()

    # TODO: save images???

    #==================================================

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
    kp1, des1 = orb.detectAndCompute(rgb_optical_original,None)
    kp2, des2 = orb.detectAndCompute(rgb_thermal_original,None)

    # Instantiate Brute Force Matcher with default parameters
    # Default Params:
    # normType = NORM_L2 (use NORM_L2 for sift/surf & NORM_HAMMING for orb, brief, brisk)
    # crossCheck = False <--- If true, only returns "mutual best match" (i best for j AND j best for i)
    bf = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)

    # Find matches with Brute Force
    matches = bf.match(des1,des2)

    # Sort matches in order of distance
    matches = sorted(matches, key = lambda x:x.distance)

    # Take only the first 'n' matches (default all when param is zero)
    if args.num_matches: matches = matches[:args.num_matches]

    # Draw matches
    opt_therm_sift_match = cv2.drawMatches(rgb_optical_original,kp1,
                                rgb_thermal_original,kp2,
                                matches,None,
                                # flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                                )

    # Show images
    cv2.imshow('Opt-Therm Match ORB - ' + str(timestamp), opt_therm_sift_match)
    cv2.waitKey()

    # TODO: save images???



if __name__ == "__main__":
    main()