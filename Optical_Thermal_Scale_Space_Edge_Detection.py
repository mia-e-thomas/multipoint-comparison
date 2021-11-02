# External Imports
from h5py._hl import dataset
import numpy as np
import cv2 
from pathlib import Path            # Absolute path to dataset
import os                           # Overwrite folder
import shutil                       # Overwrite folder
import h5py                         # Extract dataset
from scipy.ndimage.filters import * # Gaussian blur for scale space
import skimage.measure              # Downsampling
#import matplotlib.pyplot as plt

# My functions
from mp_plotting import *

##############
# Parameters #
##############
# TODO: handle this with yaml

############## CHANGE THIS PARAMETER TO GET DIFF IMAGE ##############
# Image to be evaluated
key = 900

# Good (and bad) examples
# Good:
# - 415 (Octave 2 Blur 1 & Octave 3 blur 2 & Octave 4 blur 2)
# - 430 (Octave 2 Blur 1 & Octave 4 blur 1)
# - 20  (Octave 0 Blur 2 & Octave 1 blur 1 & Octave 2 blur 1 & more)
# - 500 (Octave 2, 3, 4)
# - 600 (Octave 1)
# - 700 (Octave 4 Blur 1)
# - 900 (Octave 2 Blur 1, 2 & Octave 3 Blur 2)
#
# Bad: 
# - 445 
# - 460 
# - 1
# - 10
# - 30
# - 800

# Folders
dataset_type = 'test'
dataset_path = str(Path.home()) + "/MASc_Research/datasets/eth_multispect_im_pair/multipoint/data/"+ dataset_type + ".hdf5" # Path to dataset
results_path = str(Path.home()) + "/MASc_Research/datasets/eth_multispect_im_pair/analyze_multipoint_mt/image_results/"     # Path to resulting images

# Edge detector parameters
canny_thres_1_base = 75
canny_thres_2_base = 175
canny_step = 10
kernel_sz = 3

# Scale Space Parameters
octave_levels = 5          # Downsample by factor of 2 each time
scale_levels  = 5          # Number of different sized Gaussian filters to apply
base_sigma    = 1.6        # Starting sigma for gauss blur
scale_sigma   = np.sqrt(2) # Factor to increase sigma of progressive gaussian blurs

# Runtime parameters
show_images = True
parameter_file = 'file_parameters.txt'

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

# Plot optical and thermal images side by side
optical_im = im_trio['optical'][:][:]
rgb_optical = cv2.cvtColor((optical_im*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

thermal_im = im_trio['thermal'][:][:]
rgb_thermal = cv2.cvtColor((thermal_im*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

opt_therm_stack_original = np.hstack((rgb_optical, rgb_thermal))
#cv2.imshow('Optical and Thermal Images: ' + timestamp, opt_therm_stack)
#cv2.waitKey()

#------- Scale Space and Save Images -------#

# 1) Make folder for timestamp
folder_name = results_path + dataset_type + '/' + timestamp + '/'
# Start with a fresh folder each time because the titles are parameter dependent
if os.path.exists(folder_name):
    shutil.rmtree(folder_name)
os.makedirs(folder_name)
# old method
#Path(folder_name).mkdir(parents=True, exist_ok=True)

# 2) Export text file with parameters
parameter_string = ("Parameters:\n" +
"\nImage Pair: \n" +
"dataset_type = " + str(dataset_type) + "\n" +
"dataset_path = " + str(dataset_path) + "\n" +
"results_path = " + str(results_path) + "\n" +
"folder_name = " + str(folder_name) + "\n" +
"key = "          + str(key) +          "\n" + 
"timestamp = "    + str(timestamp) +    "\n" + 
"\nCanny Params: \n" + 
"canny_thres_1_base = " + str(canny_thres_1_base) +"\n" + 
"canny_thres_2_base = " + str(canny_thres_2_base) +"\n" + 
"canny_step = " + str(canny_step) +"\n" + 
"kernel_sz = " + str(kernel_sz) +"\n" + 
"canny_thres_1_base = " + str(canny_thres_1_base) +"\n" + 
"canny_thres_1_base = " + str(canny_thres_1_base) +"\n" + 
"\nScale Space Params: \n" + 
"octave_levels = " + str(octave_levels) +"\n" + 
"scale_levels = " + str(scale_levels) +"\n" + 
"base_sigma = " + str(base_sigma) +"\n" + 
"scale_sigma = " + str(scale_sigma) +"\n" + 
"\nRuntime Params: \n" + 
"show_images = " + str(show_images) +"\n" + 
"parameter_file = " + str(parameter_file) +"\n")

p_file = open(folder_name + parameter_file, "a")
p_file.write(parameter_string)
p_file.close()


# 3) Outer loop for the octave
for octave in np.arange(octave_levels):
    # If octave == 0: no downsample
    if not octave:
        opt_therm_stack_sampled = opt_therm_stack_original

    # If octave =/= 0: downsample by octave
    else:
        opt_therm_stack_sampled = skimage.measure.block_reduce(opt_therm_stack_original, block_size=(octave*2,octave*2,1), func=np.mean, cval=0, func_kwargs=None).astype(np.uint8)

    # 4) Inner loop for the Gaussian blurs
    for blur_level in np.arange(scale_levels):
        # 4a) Gauss blur
        # If blur level == 0 => leave image untouched
        if not blur_level:
            opt_therm_stack_blur = opt_therm_stack_sampled
        
        # If blur level =/= 0 => sigma = (base_sigma)^(blur_level-1)
        else:
            opt_therm_stack_blur = gaussian_filter(opt_therm_stack_sampled, sigma=base_sigma*(scale_sigma)**(blur_level-1))

        # Show images (if parameter set)
        if show_images:
            cv2.imshow('Optical-and-Thermal-Processed'+ '_Octave-' + str(octave) + '_Blur-level-' + str(blur_level), opt_therm_stack_blur)
            cv2.waitKey()

        # Write image
        cv2.imwrite(folder_name + 'Optical-and-Thermal-Processed'+ '_Octave-' + str(octave) + '_Blur-level-' + str(blur_level) + '_Timestamp-' + str(timestamp) + '.png', opt_therm_stack_blur)

        # 4b) Canny detector 
        # set thresholds
        canny_thres_1 = canny_thres_1_base - canny_step*blur_level
        canny_thres_2 = canny_thres_2_base - canny_step*blur_level
        opt_therm_stack_edges = cv2.Canny(image=opt_therm_stack_blur, threshold1 = canny_thres_1, threshold2=canny_thres_2, apertureSize=kernel_sz)

        # Show images (if parameter set)
        if show_images:
            cv2.imshow('Optical-and-Thermal-Processed'+ '_Octave-' + str(octave) + '_Blur-level-' + str(blur_level) + '_Edges_Canny1-' + str(canny_thres_1) + '_Canny2-' + str(canny_thres_2), opt_therm_stack_edges)
            cv2.waitKey() 

        # Write image
        cv2.imwrite(folder_name + 'Optical-and-Thermal-Processed'+ '_Octave-' + str(octave) + '_Blur-level-' + str(blur_level)
        + '_Edges_Canny1-' + str(canny_thres_1) + '_Canny2-' + str(canny_thres_2) + '_Timestamp-' + str(timestamp) + '.png', opt_therm_stack_edges)



