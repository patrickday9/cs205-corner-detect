from __future__ import division
import pyopencl as cl
import numpy as np
from PIL import Image
import pylab
from scipy.ndimage import filters
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy import linalg
import time
import sys

# Call functions from serial implementation code, we use these functions for time comparisons 
# and for displaying the output of our openCL GPU code
from harris import Timer, compute_harris_response, run_harris, plot_harris_points, round_up
from harris import check_dim, get_harris_points, plot_harris_points, generate_weights
from harris import appendimages, plot_matches, match_twosided, match, get_descriptors
from harris import serial_matching

# Call parallel functions for final harris corner dection algorthim we wrote
from harris_corner_detection_final_driver import harris_get_corners

# Call parallel corner matching function
from parallel_matching_v1 import parallel_matching

# Call parallel corner matching and sort function
from parallel_matching_final import parallel_match_sort

if __name__ == '__main__':

	''' Import both images and the number of iterations'''
	if len(sys.argv) > 3: 
		image1 = sys.argv[1]
		image2 = sys.argv[2]
		num_runs = int(sys.argv[3])
		print
		print "===================================================================================="
		print "Running Harris Corner detection for", image1, "and", image2
		print "For the following number of iterations:", num_runs

	else:
		print 'No image specified'
		print 'Run as follows: serial_mapping.py <image1.png> <image2.jpg>\n'

	# Run Harris Get Corners for Image 1
	Harris_Matrix_1, serial_points_1, parallel_points_1, output_times_openCL_1, \
		output_times_serial_1 = harris_get_corners(num_runs, image1)

	# Run Harris Get Corners for Image 1
	Harris_Matrix_2, serial_points_2, parallel_points_2, output_times_openCL_2, \
		output_times_serial_2  = harris_get_corners(num_runs, image2)			

	# Run Parallel Matching and Sort with Parallel Harris corners
	match_parallel = parallel_match_sort(Harris_Matrix_1, Harris_Matrix_2, \
		parallel_points_1, parallel_points_2)

	# Run Parallel Matching and Sort with Parallel Harris corners
	match_serial = serial_matching(Harris_Matrix_1, Harris_Matrix_2, \
		serial_points_1, serial_points_2)

	# Plot Parallel Matches
	plot_matches(image1, image2, parallel_points_1, parallel_points_2, \
		match_parallel, 'parallel_match.jpg')

	# Plot Serial Matches
	plot_matches(image1, image2, serial_points_1, serial_points_2, \
		match_serial, 'serial_match.jpg')
