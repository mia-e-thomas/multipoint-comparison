import numpy as np
import cv2 
import os                                
import shutil                            
import h5py                              
import yaml
import argparse
import matplotlib.pyplot as plt
import math
import time

# Local packages
from mfd.src.feature_matching import FeatureMatching
from mfd.src.mfd import MFD

import matplotlib
matplotlib.use('TKAgg')

# Set MAXIMUM number of features to bound matching time
max_features = 10000

# Nearest neighbour ratio
nnr = 0.8

def main(): 

    #------------#
    # Parameters #
    #------------#
    # Define script arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--yaml-config', default='config/config_feature_homography.yaml', help='YAML config file')
    parser.add_argument('-i', '--index', default=500, type=int, help='Index of image pair')
    parser.add_argument('-s', dest='show', action='store_true', help='If set, the prediction the images/plots are displayed')

    # Get arguments
    args = parser.parse_args()

    # Get YAML file
    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Configure paths
    dataset_path = os.getcwd() + '/' + config['dir_dataset']
    results_path = os.getcwd() + '/' + config['dir_results']

    # Suppress scienfitic notation for numpy
    np.set_printoptions(suppress=True)

    # Manually set parameter
    hist_bin_size = 10

    #----------------#
    # Data & Folders # 
    #----------------#

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

    #------ Make folder for specific image pair ------#
    folder_name = results_path + '/' + timestamp + '/'

    # Overwrite existing folder
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)

    #------ Make output results folder ------#
    output_file = folder_name + 'Output-' + str(timestamp)

    # =====================================================================

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

    # Save individual images
    cv2.imwrite(folder_name+"Single_Img_Opt-"  +str(timestamp) +".png", rgb_optical_original)
    cv2.imwrite(folder_name+"Single_Img_Therm-"+str(timestamp) +".png", rgb_thermal_original)

    # Stack
    opt_therm_stack_original = np.hstack((rgb_optical_original,rgb_thermal_original))

    # Show
    if args.show:
        cv2.imshow('Optical (L) & Thermal (R) - ' + str(timestamp), opt_therm_stack_original)
        cv2.waitKey()
        cv2.destroyWindow('Optical (L) & Thermal (R) - ' + str(timestamp))
    
    # Save
    cv2.imwrite(folder_name+"Opt_Therm_Stacked-"+str(timestamp) +".png", opt_therm_stack_original)


    # =====================================================================


    #------#
    # SIFT # 
    #------#

    #------- TIMING -------#
    start_sift = time.process_time()

    #------- SIFT -------#
    # Instantiate SIFT detector
    # Default Params:
    # nfeatures = 0
    # nOctaveLayers = 3
    # contrastThreshold = 0.04
    # edgeThreshold = 10
    # sigma = 1.6
    # sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10) # DEFAULT PARAMS
    sift = cv2.SIFT_create(contrastThreshold=0.12, edgeThreshold=20, # MODIFIED PARAMS
                           nfeatures = max_features, # SET MAX features to bound matching time
                           )

    # Find SIFT keypoints and descriptors
    kp1_sift, des1_sift = sift.detectAndCompute(rgb_optical_original,None)
    kp2_sift, des2_sift = sift.detectAndCompute(rgb_thermal_original,None)

    ''' Method 1'''
    # Instantiate Brute Force Matcher with default parameters
    # Default Params:
    # normType = NORM_L2 (use NORM_L2 for sift/surf & NORM_HAMMING for orb, brief, brisk)
    # crossCheck = False <--- If true, only returns "mutual best match" (i best for j AND j best for i)
    bf_sift = cv2.BFMatcher(crossCheck = True)

    # Find matches with Brute Force
    matches_sift = bf_sift.match(des1_sift,des2_sift)

    # Sort matches in order of distance
    matches_sift = sorted(matches_sift, key = lambda x:x.distance)
    ''' End Method 1 '''
    
    ''' Method 2: Knn Match'''
    '''
    bf_sift = cv2.BFMatcher_create(normType=cv2.NORM_L2)
    matches_sift = bf_sift.knnMatch(des1_sift,des2_sift,k=2)

    matches_sift = FeatureMatching.nearest_neighbor_test(matches_sift,nnr)
    '''
    ''' End Method 2 '''

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
    cv2.destroyWindow('Opt-Therm Match SIFT - ' + str(timestamp))

    # Save Image
    cv2.imwrite(folder_name+"Opt_Therm_Match_SIFT-"+str(timestamp) +".png", opt_therm_match_sift)

    #------- TIMING -------#
    time_sift = time.process_time() - start_sift
    print("SIFT time (s): " + str(time_sift))


    # =====================================================================

    #-----#
    # ORB # 
    #-----#

    #------- TIMING -------#
    start_orb = time.process_time()

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
    orb = cv2.ORB_create(edgeThreshold = 31, patchSize = 31, fastThreshold= 20,
                         nfeatures = max_features, # SET MAX features to bound matching time
                         )

    # Find keypoints and descriptors
    kp1_orb, des1_orb = orb.detectAndCompute(rgb_optical_original,None)
    kp2_orb, des2_orb = orb.detectAndCompute(rgb_thermal_original,None)

    ''' Matching Method 1'''
    # Instantiate Brute Force Matcher with default parameters
    # Default Params:
    # normType = NORM_L2 (use NORM_L2 for sift/surf & NORM_HAMMING for orb, brief, brisk)
    # crossCheck = False <--- If true, only returns "mutual best match" (i best for j AND j best for i)
    bf_orb = cv2.BFMatcher(normType = cv2.NORM_HAMMING, crossCheck = True)

    # Find matches with Brute Force
    matches_orb = bf_orb.match(des1_orb,des2_orb)

    # Sort matches in order of distance
    matches_orb = sorted(matches_orb, key = lambda x:x.distance)
    ''' End Method 1'''

    ''' Method 2: knn '''
    '''
    bf_orb = cv2.BFMatcher(normType = cv2.NORM_HAMMING)
    matches_orb = bf_orb.knnMatch(des1_orb,des2_orb,k=2)

    matches_orb = FeatureMatching.nearest_neighbor_test(matches_orb,nnr)
    '''
    ''' End Method 2'''

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
    cv2.destroyWindow('Opt-Therm Match ORB - ' + str(timestamp))

    # Save Image
    cv2.imwrite(folder_name+"Opt_Therm_Match_ORB-"+str(timestamp) +".png", opt_therm_match_orb)

    #------- TIMING -------#
    time_orb = time.process_time() - start_orb
    print("ORB time (s): " + str(time_orb))

    # =====================================================================

    #-----#
    # MFD # 
    #-----#

    #------- TIMING -------#
    start_mfd = time.process_time()

    #------- 1) Descriptor Algorithm -------#
    descriptor_algorithm = MFD()

    #------- 2) Detector Algorithm -------#
    detector_algorithm = cv2.FastFeatureDetector_create(
        # threshold=40, # ORIGINAL
        threshold=20, # MODIFIED
        nonmaxSuppression=True,
        )
   
    #------- 3) Feature Matching -------#
    # Instantiate feature matcher
    fm = FeatureMatching(detector_algorithm, descriptor_algorithm)

    # Match features given images
    opt_term_match_mfd, num_match, precision, kp1_mfd, kp2_mfd, matches_mfd = fm.match_features(
        folder_name+"Single_Img_Opt-"  +str(timestamp) +".png", 
        folder_name+"Single_Img_Therm-"+str(timestamp) +".png", 
        )
    
    # Show image
    cv2.imshow('Opt-Therm Match MFD - ' + str(timestamp), opt_term_match_mfd)
    cv2.waitKey()
    cv2.destroyWindow('Opt-Therm Match MFD - ' + str(timestamp))

    # Save image
    cv2.imwrite(folder_name+"Opt_Therm_Match_MFD-"+str(timestamp) +".png", opt_term_match_mfd)


    #------- TIMING -------#
    time_mfd = time.process_time() - start_mfd
    print("MFD time (s): " + str(time_mfd))

    
    # =====================================================================

    matches_tot = [matches_sift, matches_orb, matches_mfd]
    kp1_tot     = [kp1_sift    , kp1_orb    , kp1_mfd    ]
    kp2_tot     = [kp2_sift    , kp2_orb    , kp2_mfd    ]
    name_tot    = ['SIFT'      , 'ORB'      , 'MFD'      ]

    # =====================================================================

    for index in np.arange(len(matches_tot)): 
    # for (matches, kp1, kp2) in (matches_tot,kp1_tot,kp2_tot):

        matches = matches_tot[index]
        kp1     = kp1_tot[index]
        kp2     = kp2_tot[index]
        name    = name_tot[index]

        #------- MIN # MATCHES -------#

        # Check for minimum 4 matches (for homography)
        suff_match = False
        if len(matches) > 3: suff_match = True
        
        if not suff_match:
            print("Insufficient Matches (" + str(len(matches)) + ") to compute homography")

        else: 

            #------- HOMOGRAPHY -------#
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC) # MODIFIED
            print(name + " Homography: \n" + str(M))
            
            #------- REDRAW INLIERS -------#
            # 1. Draw all matches in red
            opt_therm_match = cv2.drawMatches(rgb_optical_original,kp1,rgb_thermal_original,kp2,matches, None,
                                        matchColor = (0,0,255),       # draw matches in red color
                                        )
            # 2. Draw inliers in green 
            w = int(opt_therm_match.shape[1]/2)
            opt_therm_match = cv2.drawMatches(opt_therm_match[:,:w,:],kp1,opt_therm_match[:,w:,:],kp2,matches, None,
                                        matchColor = (0,255,0),       # draw matches in green color
                                        matchesMask = mask[:,0],  # draw only inliers
                                        )
            # Show images
            cv2.imshow(name + ' Inliers and Outliers - ' + str(timestamp), opt_therm_match)
            cv2.waitKey()
            cv2.destroyWindow(name + ' Inliers and Outliers - ' + str(timestamp))
            # Save Image
            cv2.imwrite(folder_name+"Opt_Therm_Match_" + name + "_Inliers_Outliers-"+str(timestamp) +".png", opt_therm_match)


            #------- APPLY HOMOGRAPHY -------#
            # Compute
            opt_warped = cv2.warpPerspective(rgb_optical_original, M, rgb_optical_original.T.shape[1:3])
            # Show
            cv2.imshow('Optical Warped '+name+' - ' + str(timestamp), opt_warped)
            cv2.waitKey()
            cv2.destroyWindow('Optical Warped '+name+' - ' + str(timestamp))
            # Save
            cv2.imwrite(folder_name+'Optical_Warped_'+name+'-'+str(timestamp)+'.png', opt_warped)

            #-------#
            # ERROR # 
            #-------#
            # Pixel location in source and destination should be THE SAME
            # Error Per Pixel: Euclidean distance b/w source & dest pixel locations
            # Overall Error: Average

            #------- 1) ALL MATCHES -------#
            # a) Get euclidean distance b/w corresponding src & dst
            err_pts = np.linalg.norm(src_pts - dst_pts,axis=2)

            # b) Take average of ALL distances for average pixel error
            err_avg = np.average(err_pts, axis=0)
            print("Average (ALL) Pixel Error "+name+": " + str(err_avg))

            #------- 2) INLIERS ONLY -------#
            # c) Take only elements where mask is nonzero
            err_pts_inlier = err_pts[mask[:,0] != 0, :]

            # d) Average inlier pixel distance 
            err_avg_inlier = np.average(err_pts_inlier, axis=0) 
            print("Average (INLIER) Pixel Error "+name+": " + str(err_avg_inlier))

            #------- 3) OUTLIERS ONLY -------#
            # e) Get outliers
            err_pts_outlier = err_pts[mask[:,0] == 0,:]

            #-----------#
            # HISTOGRAM # 
            #-----------#
            # Calculate histogram w/ and w/o mask
            
            # b) Histogram of ONLY inliers
            max_val = math.ceil(err_pts_inlier[:,0].max())
            plt.hist(err_pts_inlier[:,0], 
                        bins=max_val, range=[0,max_val],
                        edgecolor='black')
            plt.xlabel(name+" Inlier Matching Error (Euclidean Distance in Pixels)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            # Save image
            plt.savefig(folder_name+'Match_Error_Inliers_Hist_'+name+'-'+str(timestamp)+'.png')
            # Show Image
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.clf()
            # plt.close()

            # c) Stacked histogram of inliers and outliers
            num_bins = math.ceil(err_pts.max()/hist_bin_size) # set bin size to 'hist_bin_size'
            plt.hist([err_pts_inlier[:,0],err_pts_outlier[:,0]], 
                        bins=num_bins, stacked=True,
                        color=["g","r"], # green inliers red outliers
                        label=['Inliers','Outliers'],
                        edgecolor='black',
                    )
            plt.legend(prop={'size': 10})
            plt.xlabel(name+" Matching Error (Euclidean Distance in Pixels)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            # Save image
            plt.savefig(folder_name+'Match_Error_Stacked_Hist_'+name+'-'+str(timestamp)+'.png')
            # Show Image
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.clf()
            # plt.close()


            #------- PRINT OTHER STATS -------#
            print("Total Matches  "+name+": " + str(err_pts.shape[0]))
            print("Total Inliers  "+name+": " + str(err_pts_inlier.shape[0]))
            print("Total Outliers "+name+": " + str(err_pts_outlier.shape[0]))
            print("---------------------------")

    # =====================================================================

if __name__ == "__main__":
    main()