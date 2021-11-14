# External Imports
from   h5py._hl import dataset
import numpy as np
import cv2 
from   pathlib import Path               # Absolute path to dataset
import os                                # Overwrite folder
import shutil                            # Overwrite folder
import h5py                              # Extract dataset  
from   scipy.ndimage.filters import *    # Gaussian blur for scale space
from   scipy import ndimage              # Used for mutual info
import skimage.measure                   # Downsampling
from   sklearn.metrics.cluster import *  # Compute mutual information
import csv

#from phasepack import phasecongmono # Phase congruency
#import matplotlib.pyplot as plt

# My functions
from Optical_Thermal_Comp_Support import *

##############
# Parameters #
##############
# TODO: handle this with yaml instead of CSV

#------------- Most important Params -------------#
key = 500             # Image pair to be evaluated
show_images = True    # Show images in script or no
window_size = 3       # Window size for correlation

#------------- All Other Params -------------#
# Folders
dataset_type = 'test' # test or training
dataset_path = str(Path.home()) + "/masc-research/datasets/eth_multispect_im_pair/multipoint/data/"+ dataset_type + ".hdf5" # Path to dataset
results_path = str(Path.home()) + "/masc-research/datasets/eth_multispect_im_pair/analyze_multipoint_mt/image_results/"     # Path to resulting images

# Edge detector parameters
canny_thres_1_base = 75   # Lower Canny threshold
canny_thres_2_base = 175  # Upper Canny threshold
canny_step = 10           # *** Amount to *decrease* canny thresholds every time blur level is increased ***
kernel_sz = 3             # Kernel size for canny function

# Scale Space Parameters
octave_levels = 5          # Downsample by factor of 2 each octave (octave 0 means no downsample)
scale_levels  = 5          # Number of different sized Gaussian filters to apply in each octave (scale 0 means no blur)
base_sigma    = 1.6        # Starting sigma for gauss blur
scale_sigma   = np.sqrt(2) # Factor to increase sigma of progressive gaussian blurs

# Runtime parameters
csv_file       = 'image_results.csv'


#########
# Start # 
#########

#------- Get image pair -------#
# Open hdf5 file
f = h5py.File(dataset_path, 'r')

# Get the keys (each key is a different timestep with all 3 images)
timestamp_keys = list(f.keys())

# Get the image types 
im_types = list(f[timestamp_keys[0]].keys())

# Select the timestamp and get the image trio
timestamp = timestamp_keys[key]
im_trio   = f[timestamp]

#------- Stack thermal and optical images together -------#

# Get RGB image and convert to uint8
optical_im = im_trio['optical'][:][:]
rgb_optical_original = cv2.cvtColor((optical_im*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

# Get thermal image and convert to uint8
thermal_im = im_trio['thermal'][:][:]
rgb_thermal_original = cv2.cvtColor((thermal_im*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

#------- Scale Space and Analysis -------#

# ------ 1) Make folder for timestamp & start CSV file ------ #
folder_name = results_path + dataset_type + '/' + timestamp + '/'

# Start with a fresh folder each time because the titles are parameter dependent
if os.path.exists(folder_name):
    shutil.rmtree(folder_name)
os.makedirs(folder_name)

# initialize CSV file
f_csv = open(folder_name+csv_file,'w')
writer = csv.writer(f_csv)
writer.writerow(['Downsample/Octave', 'Blur_Level', 'Actual_Blur', 'MI', 'NMI', 'CC', 'NCC'])

# ------ 2) Outer loop for the octave ------ #
for octave in np.arange(octave_levels):

    # If octave == 0: no downsample
    if not octave:
        rgb_optical_sampled = rgb_optical_original
        rgb_thermal_sampled = rgb_thermal_original

    # If octave =/= 0: downsample by octave
    else:
        rgb_optical_sampled = skimage.measure.block_reduce(rgb_optical_original, block_size=(octave*2,octave*2,1), func=np.mean, cval=0, func_kwargs=None).astype(np.uint8)
        rgb_thermal_sampled = skimage.measure.block_reduce(rgb_thermal_original, block_size=(octave*2,octave*2,1), func=np.mean, cval=0, func_kwargs=None).astype(np.uint8)

    # ------ 3) Inner loop for the Gaussian blurs ------ #
    for blur_level in np.arange(scale_levels):

        # ------ 3a) Gauss blur ------ #
        # If blur level == 0 => leave image untouched
        if not blur_level:
            rgb_optical_blur = rgb_optical_sampled
            rgb_thermal_blur = rgb_thermal_sampled
        # If blur level =/= 0 => sigma = (base_sigma)^(blur_level-1)
        else:
            rgb_optical_blur = gaussian_filter(rgb_optical_sampled, sigma=base_sigma*(scale_sigma)**(blur_level-1))
            rgb_thermal_blur = gaussian_filter(rgb_thermal_sampled, sigma=base_sigma*(scale_sigma)**(blur_level-1))


        # ------ 3b) Cross-correlation ------ #
        x_corr      = cv2.matchTemplate(rgb_optical_blur,rgb_thermal_blur, cv2.TM_CCORR).item()
        x_corr_norm = cv2.matchTemplate(rgb_optical_blur,rgb_thermal_blur, cv2.TM_CCORR_NORMED).item()

        # ------ 3c) Mutual Information ------ #
        mi      =            mutual_info_score(rgb_optical_blur[:,:,0].ravel(),rgb_thermal_blur[:,:,0].ravel())
        mi_norm = normalized_mutual_info_score(rgb_optical_blur[:,:,0].ravel(),rgb_thermal_blur[:,:,0].ravel())

        # ------ 3d) Write parameters to CSV ----- #
        blur = (base_sigma*(scale_sigma)**(blur_level-1)) if blur_level else blur_level # Calculate actual blur
        writer.writerow([octave, blur_level, "%.4f" % blur, "%.4f" % mi, "%.4f" %  mi_norm, "%.4f" %  x_corr, "%.4f" % x_corr_norm])

        # ------ 3e) Stack, show, save images ------ #
        # Stack
        opt_therm_stack_blur = np.hstack((rgb_optical_blur, rgb_thermal_blur))
        # Format string
        opt_therm_blur_string = "Optical-and-Thermal-Processed_Octave-%d_Blur-level-%.2f_NMI-%.3f_NCC-%.3f" % (octave,blur_level,mi_norm,x_corr_norm) 
        # Show images (if parameter "show_images" is True)
        if show_images:
            cv2.imshow(opt_therm_blur_string, opt_therm_stack_blur)
            cv2.waitKey()
        # Write image pair
        cv2.imwrite(folder_name + opt_therm_blur_string + '_Timestamp-' + str(timestamp) + '.png', opt_therm_stack_blur)

        # ------ 3f) Canny detector ------ #
        # set thresholds
        canny_thres_1 = canny_thres_1_base - canny_step*blur_level
        canny_thres_2 = canny_thres_2_base - canny_step*blur_level
        # Run edge detector
        rgb_optical_edges = cv2.Canny(image=rgb_optical_blur, threshold1 = canny_thres_1, threshold2=canny_thres_2, apertureSize=kernel_sz)
        rgb_thermal_edges = cv2.Canny(image=rgb_thermal_blur, threshold1 = canny_thres_1, threshold2=canny_thres_2, apertureSize=kernel_sz)
        # Stack edge images
        opt_therm_stack_edges = np.hstack((rgb_optical_edges, rgb_thermal_edges))
        # Format string
        opt_therm_edge_string = "Optical-and-Thermal-Processed_Octave-%d_Blur-level-%.2f_Edges_Canny1-%d_Canny2-%d" % (octave,blur_level,canny_thres_1,canny_thres_2) 
        # Show images (if parameter set)
        if show_images:
            cv2.imshow(opt_therm_edge_string, opt_therm_stack_edges)
            cv2.waitKey() 
        # Write image pair
        cv2.imwrite(folder_name + opt_therm_edge_string + '_Timestamp-' + str(timestamp) + '.png', opt_therm_stack_edges)

        # ------ 3g) Phase Congruency ------ #
        '''
        (PC_opt,   OR, ft, T) = phasecongmono(rgb_optical_blur[:,:,0])
        (PC_therm, OR, ft, T) = phasecongmono(rgb_thermal_blur[:,:,0])
        opt_therm_stack_PC = np.hstack((PC_opt,PC_therm))
        cv2.imshow('PC'+ '_Octave-' + str(octave) + '_Blur-level-' + str(blur_level), opt_therm_stack_PC)
        cv2.waitKey()
        '''

        # ------ 3h) Whole-image Correlation ------ #  
        '''
        # Compute normalized cross correlation
        im_corr = norm_x_correlation(rgb_optical_blur, rgb_thermal_blur, win_size = window_size)
        # Map the [-1,1] to [0,1]
        im_corr_int8 = ((im_corr+1)*255/2).astype(np.uint8)
        # Colour map to more easily visualize results
        im_corr_map  = cv2.applyColorMap(im_corr_int8, cv2.COLORMAP_JET)
        # Display Colourmap
        cv2.imshow('Colourized-Normalized-X-Correlation'+ '_Octave-' + str(octave) + '_Blur-level-' + str(blur_level) + '_Win-Size-' + str(window_size), im_corr_map)
        cv2.waitKey()
        # Display Normal
        cv2.imshow('Normalized-X-Correlation'+ '_Octave-' + str(octave) + '_Blur-level-' + str(blur_level)+ '_Win-Size-' + str(window_size), im_corr)
        cv2.waitKey()
        # Write 
        cv2.imwrite(folder_name + 'Colour-Map-Normalized-X-Correlation'+ '_Octave-' + str(octave) + '_Blur-level-' + str(blur_level)+ '_Win-Size-' + str(window_size) + '.png', im_corr_map)
        cv2.imwrite(folder_name + 'Normalized-X-Correlation'+ '_Octave-' + str(octave) + '_Blur-level-' + str(blur_level)+ '_Win-Size-' + str(window_size) + '.png', im_corr_int8)
        '''


# 4) Export text file with parameters
# Write parameters to csv file
writer.writerow(["Parameters:"])
writer.writerow(["\nImage Pair:"])
writer.writerow(["key"         ,key])      
writer.writerow(["timestamp"   ,timestamp])
writer.writerow(["\nCanny Params:"  ]) 
writer.writerow(["canny_thres_1_base",canny_thres_1_base])
writer.writerow(["canny_thres_2_base",canny_thres_2_base]) 
writer.writerow(["canny_step"        ,canny_step]) 
writer.writerow(["kernel_sz"         ,kernel_sz])
writer.writerow(["canny_thres_1_base",canny_thres_1_base])
writer.writerow(["canny_thres_1_base",canny_thres_1_base])
writer.writerow(["\nScale Space Params:"])
writer.writerow(["octave_levels"     ,octave_levels])
writer.writerow(["scale_levels"      ,scale_levels])
writer.writerow(["base_sigma"        ,base_sigma])
writer.writerow(["scale_sigma"       ,scale_sigma])
writer.writerow(["\nRuntime Params:"]) 
writer.writerow(["show_images"       ,show_images])
writer.writerow(["csv_file"          ,csv_file])
writer.writerow(["\nCorrelation Params:"]) 
writer.writerow(["window_size"       ,window_size])
# Close CSV file
f_csv.close()

