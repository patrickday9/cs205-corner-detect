import numpy as np
import pylab as py
from scipy import linalg
import sys
import pylab
from PIL import Image

from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import filters
import matplotlib
import matplotlib.pyplot as plt
import time
import math


def serial_matching(harris_im1, harris_im2, filtered_coords1, filtered_coords2):
    
    d1 = get_descriptors(harris_im1, filtered_coords1)
    d2 = get_descriptors(harris_im2, filtered_coords2)

    print 'Starting Serial Matching'
    print "Shape of Image 1:", np.shape(harris_im1)
    print "Number of Corners found in Image 1:", len(filtered_coords1)
    print "Shape of Descriptor1 Window:", np.shape(d1)
    print
    print "Shape of Image 2:", np.shape(harris_im2)
    print "Number of Corners found in Image 2:", len(filtered_coords2)
    print "Shape of Descriptor2 Window:", np.shape(d2)

    start = time.time()
    matches = match_twosided(d1, d2)
    end = time.time()

    print "Total Serial Matching Execution Time:", end - start

    return matches

def appendimages(im1, im2):
    """ Return a new image that appends the two images side-by-side. """
    
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    
    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2-rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1-rows2, im2.shape[1]))), axis=0)
    
    # if none of these cases they are equal, no filling needed.
    return np.concatenate((im1, im2), axis=1)

def plot_matches(im1, im2, locs1, locs2, matchscores, filename, show_below=False):
    np_im1 = np.array(Image.open(str(im1)).convert('L')).astype(np.float32)[::1, ::1].copy()
    np_im2 = np.array(Image.open(str(im2)).convert('L')).astype(np.float32)[::1, ::1].copy()

    im3 = appendimages(np_im1, np_im2) 
    if show_below:
        im3 = np.vstack((im3, im3))
    
    plt.figure(figsize=[12,8])
    plt.gray() 
    plt.imshow(im3, aspect='auto')

    cols1 = np_im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0: 
            plt.plot([locs1[i][1], locs2[m][1]+cols1], [locs1[i][0], locs2[m][0]], 'c')
            plt.axis('off')

    plt.savefig(filename)
   
def match_twosided(desc1,desc2,threshold=0.5):
    """ Two-sided symmetric version of match(). """
    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)
    ndx_12 = np.where(matches_12 >= 0)[0]
    
    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1 
            
    return matches_12

def match(desc1, desc2, threshold=0.5):
    """ For each corner point descriptor in the first image,
    select its match to second image using
    normalized cross correlation. """
    
    n = len(desc1[0])
    # pair-wise distances
    d = -np.ones((len(desc1),len(desc2))) 
    
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j]) 
            ncc_value = sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value
    
    ndx = np.argsort(-d)
    matchscores = ndx[:,0] 
    
    return matchscores


def get_descriptors(image, filtered_coords, wid=5):
    """ For each point return pixel values around the point
    using a neighbourhood of width 2*wid+1. (Assume points are
    extracted with min_distance > wid). """
    
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1, 
                      coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)
    
    return desc


class Timer(object):
    ''' This object is used for timing 
    analysis of functions
    '''
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


def compute_harris_response(im, sigma=1):
    """ 
    Compute the Harris corner detector response function
    for each pixel in a graylevel image. This function has 
    the proprer boundary conditions consistant with out 
    parallel implementation and is used throughout this analysis
    """
    
    # derivatives
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx, mode = 'nearest')
    #     print (imx)
    
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy, mode = 'nearest')
    #     print (imy)
    
    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx,sigma, mode = 'nearest') 
    Wxy = filters.gaussian_filter(imx*imy,sigma, mode = 'nearest') 
    Wyy = filters.gaussian_filter(imy*imy,sigma, mode = 'nearest')
    
    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    

    return Wdet / Wtr

def compute_harris_response_original(im, sigma=1):
    """ Compute the Harris corner detector response function
        for each pixel in a graylevel image. This is the original 
        furntion, which uses the 'reflection' boundary condition"""
    
    # derivatives
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    #     print (imx)
    
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)
    #     print (imy)
    
    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx*imx,sigma) 
    Wxy = filters.gaussian_filter(imx*imy,sigma) 
    Wyy = filters.gaussian_filter(imy*imy,sigma)
    
    # determinant and trace
    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy
    
    return Wdet / Wtr

def get_harris_points(harrisim, min_dist=10, threshold=0.1): 
    """ Return corners from a Harris response image
    min_dist is the minimum number of pixels separating
    corners and image boundary. """
    
    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    
    # get coordinates of candidates
    coords = np.array(harrisim_t.nonzero()).T # ...and their values
    candidate_values = [harrisim[c[0],c[1]] for c in coords] # sort candidates
    index = np.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = [] 
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i]) 
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                              (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0 
    
    return filtered_coords


def plot_harris_points(image,filtered_coords, im_name = 'Harris_Serial_Image'):
    """ Plots corners found in image. """
    
    plt.figure(figsize=[12,8])
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'ro') 
    plt.axis('off')
    plt.title(str(im_name))
    plt.savefig(str(im_name))

def check_dim(im):
    '''this is a helper function to check 
    the dimensions of images for consistancy'''


    if len(im.shape) > 2:

        return im[:, :, 1]
    
    else:
        return im

def run_harris(im, im_name = 'Harris_Serial_Image'):
    ''' this is a wraper function to run the serial
    Harris Corner detection algorithm'''

    im_1dim = check_dim(im)
    harris_im = compute_harris_response(im_1dim)
    filtered_coords = get_harris_points(harris_im, 10)
    return filtered_coords

def round_up(global_size, group_size):
    '''This function rounds up, used
    for global size function'''

    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

def generate_weights(sigma, order = 1):
    '''This function generates the gaussian weights
    vector of a specified order, ref from Scipy.org 
    '''

    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    truncate = 4.0
    lw = int(truncate * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    if order == 1:
        weights[lw] = 0.0
        for ii in range(1, lw + 1):
            x = float(ii)
            tmp = -x / sd * weights[lw + ii]
            weights[lw + ii] = -tmp
            weights[lw - ii] = tmp

    return weights

if __name__ == '__main__':
  
    pt_x = 0
    pt_y = 200
    numpy_image = np.array(Image.open('1.tif').convert('L'))
    response = compute_harris_response(numpy_image, sigma=1)
    serial_points = get_harris_points(response, min_dist=10, threshold=0.1)
    print 'true Harris Matrix', len(serial_points)

    with Timer() as t:
        harris = run_harris(numpy_image)

    
    print t.interval



