# cs205-corner-detect
## CS205 Final Project Parallel Corner and Matching 

## Final Project: Parallel Corner Detection and Matching

## Group Members: Patrick Day and Patrick Kuiper

In order to execute our program, you must have the following files in your working directory:

1) "1.tif" - the first image

2) "2.tif" - the second image, similar to the first but taken from a different perspective.

3) "harris.py" - This file includes a number of support functions which are implemented serially. Most of these functions were taken from "Programming Computer Vision with Python" by Jan Erik Solem. These functions are used for two reasons. First, we use Solem's full serial implementation of the Harris Corner detection (which uses the Scipy library) and serial point matching functions as a baseline. These expertly developed serial implementations are used to compare the results of our parallel openCL code (both corner detection and matching). Second, we use several of these basic serial functions for non-computationally dense portions of our work. 

4) "harris_coner_detection_final_driver.py" - This is the final parallel corner detection python driver, which performed optimally. Profiling stats are provided with comparison to serial implementation. This will save two images to your directory, "Harris openCL Image.png" and "Harris Serial Image.png." To run, specify the image and number of iterations for timing profiling by running:

python harris_coner_detection_final_driver.py [image.tif] [num_runs] (uses '1.tif' with 100 iterations runs by default)

5) "harris_coner_detection_final.cl" - This is the final parallel corner detection openCL kernels, which performed optimally. This is called by harris_coner_detection_final_driver.py. 

6) "parallel_matching_final.py" - This is the final parallel corner matching python driver, which performed optimally. This will produce an image, which is automatically saved to the working directory as "matching_final.jpg." Profiling stats are provided with comparison to serial implementation. To run, specify the image and number of iterations for timing profiling by running:

### $ python parallel_matching_final.py <imgage1.tif> <imgage2.tif> <num_runs>  ('1.tif' and '2.tif' and num_runs = 1 by default).

7) "match_pp_online_sort_v5.cl" - This is the final parallel corner detection openCL kernel, which performed optimally. This includes a parallel sorting kernel. This is called by parallel_matching_final.py.

8) "driver.py" - This is the final driver which runs all of the sub-functions; both the corner detection and corner matching together and produces the final serial and parallel images, with matching corners connected between the images. These images will be saved to your directory as "serial_match.jpg" and "parallel_match.jpg."

### $ python driver.py <imgage1.tif> <imgage2.tif> <num_runs> (no default values provided, user must specify all input values)

9) "plot_times.py" - This will plot the execution times, comparing the performance of our various versions of corner detection matching versus serial implementation. The performance data is hard coded into this file. The plot will be saved to your directory as "corner_detection_performance.pdf" and "corner_matching_performance.pdf." Once again, we compared the performance of the corner detection and corner matching separately in order to maintain resolution on performance between versions and comparison to the original serial implementation. 

### $ python plot_times.py

Execution of overall program (executes both our parallel implementations of corner detection and corner matching, with serial implementation for comparison):

With these files in your directory, execute the following command:

### $ python driver.py 1.tif 2.tif 100
