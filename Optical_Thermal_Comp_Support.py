import numpy as np

#---- Normalized x_correlation -----#
# Inputs:
# * im1: "Template"
# * im2: "Window"
# * win_size: Window size => must be odd
#
# Returns:
# * im_corr: Pixel-by-Pixel correlation
#
def norm_x_correlation(im1, im2, win_size):
    # Initialize output image size
    im_corr = np.zeros(shape = im1.shape)

    # 1) Pad Zeros according to window size
    # Check window size odd
    if (win_size % 2) != 1:
        raise ValueError('Window size must be odd')
    # Pad size
    num_pad = int(win_size/2)
    # Pad im1
    im1_pad = np.pad(im1,num_pad,mode = 'constant')
    # Pad im2
    im2_pad = np.pad(im2,num_pad,mode = 'constant')

    # 2) Outer loop for row
    for row in (np.arange(im1.shape[0]) + num_pad):

        # 3) Inner loop for columns
        for col in (np.arange(im1.shape[1]) + num_pad):

            # Create template and window (size: window x window)
            template = im1_pad[row-num_pad:row+num_pad + 1,col-num_pad:col+num_pad +1]
            window   = im2_pad[row-num_pad:row+num_pad + 1,col-num_pad:col+num_pad +1]

            # Flatten
            temp_vect = template.flatten()
            win_vect  =   window.flatten()

            # Normalize
            # a) Template
            # special case for all zeros
            if np.linalg.norm(temp_vect) == 0.0:
                temp_norm = temp_vect
            # General case: subtract mean and normalize vector
            else:
                temp_norm = (temp_vect - np.mean(temp_vect))/np.linalg.norm(temp_vect - np.mean(temp_vect))

            # b) Window
            # special case for all zeros
            if np.linalg.norm(win_vect) == 0.0:
                win_norm = win_vect
            # General case: subtract mean and normalize vector
            else:
                win_norm  = ( win_vect - np.mean( win_vect))/np.linalg.norm( win_vect - np.mean( win_vect))

            # Dot product
            x_corr = np.dot(temp_norm,win_norm)

            # Output image
            im_corr[row-num_pad,col-num_pad] = x_corr

    return im_corr

'''
#---- Mutual Information -----#
# Inputs:
# * im1, im2: two images (of the same size) 
# * base: 
#
# Returns:
# * mi: (float) mutual information number
#  
def image_mi(im1, im2, win_size):
    # Initialize output image size
'''