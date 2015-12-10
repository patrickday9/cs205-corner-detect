from __future__ import division
import pyopencl as cl
import numpy as np
import Image
from PIL import Image
from skimage import color
import pylab
from scipy.ndimage import filters
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy import linalg
import sys


# Call functions from serial implementation code, we use these functions for time comparisons 
# and for displaying the output of our openCL GPU code
from harris import Timer, compute_harris_response, run_harris, plot_harris_points, round_up
from harris import check_dim, get_harris_points, plot_harris_points, generate_weights
import time



if __name__ == '__main__':


    ''' Import the image and the number of iterations'''
    if len(sys.argv) > 2: 
        image = str(sys.argv[1])
        num_runs = int(sys.argv[2])
        print
        print "==================================================="
        print "Running Harris Corner detection for", image
        print "For the following number of iterations for profiling:", num_runs

    else:
        image = '1.tif'
        num_runs = 100
        print 'Default image:', image
        print 'Default number of iterations for profiling:', num_runs

    #Define the number of runs to get average of run times
    output_times_openCL = np.zeros(num_runs)
    output_times_serial = np.zeros(num_runs)

    #Initalize loop to get average of times
    for i in range(num_runs):
        # List our platforms
        platforms = cl.get_platforms()



        # Create a context with all the devices
        devices = platforms[0].get_devices()
        context = cl.Context(devices)

        # Create a queue for transferring data and launching computations.
        # Turn on profiling to allow us to check event times.
        queue = cl.CommandQueue(context, context.devices[0],
                                properties=cl.command_queue_properties.PROFILING_ENABLE)
        program = cl.Program(context, open('harris_corner_detection_v5.cl').read()).build(options='')

       
        #Load in image to be analyzed
        host_image = np.array(Image.open(image).convert('L')).astype(np.float32)[::1, ::1].copy()

        #start time after image load for consistancy
        start = time.time()
     
        sigma = 1 #Define the standard deviation for the gauussian
        #Generate the 1D first dimensional gaussian kernel
        filter_kernel_derivative = np.asarray(generate_weights(sigma), order = 1).astype(np.float32)
        #Generate the 1D zero derivative gaussian kernel
        filter_kernel_zero = np.asarray(generate_weights(sigma, order = 0)).astype(np.float32)
        #Determine the length of the entire weight vector based on the sigma of the gaussian
        weight_length = len(filter_kernel_derivative) #should be 9 with sigma = 1
        #This is the number of neighbors for each analyzed pixel, should be even number
        window = (weight_length - 1) #window is 8
        #the halo is the number of nieghbors on each side of the analyzed pixelsd
        halo = np.int32(window / 2.)



        host_image_filtered = np.zeros_like(host_image)
        gpu_image_in = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
        zero_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
        first_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
        derivative_kernel_x = cl.Buffer(context, cl.mem_flags.READ_WRITE, filter_kernel_derivative.size * 4)
        zero_kernel = cl.Buffer(context, cl.mem_flags.READ_WRITE, filter_kernel_zero.size * 4)

        Harris_Matrix = np.zeros_like(host_image)

        # Intermediate storage area, between Derivative of Gaussian and Gaussian Filter

        local_size = (int(halo), int(halo)) # 2D local_size
        global_size = tuple([round_up(g, l) for g, l in zip(host_image.shape[::-1], local_size)]) # shape

        width = np.int32(host_image.shape[1])
        height = np.int32(host_image.shape[0])

        local_memory = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
        local_buffer_zero_1 = cl.LocalMemory(4 * (np.shape(filter_kernel_zero)[0]))   
        local_buffer_first_1 = cl.LocalMemory(4 * (np.shape(filter_kernel_derivative)[0]))   	

        buf_width = np.int32(local_size[0] + window)
        buf_height = np.int32(local_size[1] + window)

        cl.enqueue_copy(queue, gpu_image_in, host_image, is_blocking=False)
        cl.enqueue_copy(queue, derivative_kernel_x, filter_kernel_derivative, is_blocking=False)
        cl.enqueue_copy(queue, zero_kernel, filter_kernel_zero, is_blocking=False)

    ########################################### First Kernel ##################################
    #                          This Kernel takes the first derivative of a guasisan           #
    #                  of the image in the y-direction (axis = 0) and zero Derivatives        #
    #                         of a gaussian in the y-direction (axis = 0)                     #
    ########################################### First Kernel ##################################

        #Execute Derivative of Gaussian Function
        program.gaussian_first_axis(

                            queue, global_size, local_size,
                            gpu_image_in, 
                            zero_derivative_out, 
                            first_derivative_out, 
                            local_memory, width, 
                            height, buf_width, buf_height, halo, 
                            derivative_kernel_x, zero_kernel,
                            local_buffer_first_1, local_buffer_zero_1
                            )


    ########################################### Second Kernel ##################################
    #                          This Kernel takes the first derivative of a guasisan            #
    #                  of the image in the x-direction (axis = 1) and zero Derivatives         #
    #                         of a gaussian in the x-direction (axis = 1)                      #
    ########################################### Second Kernel ##################################

        
        #allocate local memory buffers for the two filters used in the second kernel
        local_memory_axis2_1 = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
        local_memory_axis2_2 = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
        local_buffer_zero_2 = cl.LocalMemory(4 * (np.shape(filter_kernel_zero)[0]))   
        local_buffer_first_2 = cl.LocalMemory(4 * (np.shape(filter_kernel_derivative)[0]))   
        #allocate memory for the output of the second kernel 
        gpu_image_Wxx_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
        gpu_image_Wyy_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
        gpu_image_Wxy_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)

        #Execute Derivative of Gaussian Function
        program.gaussian_second_axis(

                            queue, global_size, local_size, 
                            zero_derivative_out, 
                            first_derivative_out, 
                            gpu_image_Wxx_derivative_out, 
                            gpu_image_Wyy_derivative_out, 
                            gpu_image_Wxy_derivative_out,
                            local_memory_axis2_1, local_memory_axis2_2,
                            width, 
                            height, buf_width, buf_height, halo, 
                            derivative_kernel_x, zero_kernel,
                            local_buffer_first_2, local_buffer_zero_2

                            )

    ########################################### Third Kernel ###################################
    #                          This Kernel applies a gaussian to the product of the            #
    #                          parital derivatives in the y-direction (axis = 0)               #
    #                                                                                          #
    ########################################### Third Kernel ###################################


        #load in the local memory buffer allocation for all the compents fo the harris matrix
        local_memory_filter_Wxx = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
        local_memory_filter_Wyy = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
        local_memory_filter_Wxy = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
        local_buffer_zero_3 = cl.LocalMemory(4 * (np.shape(filter_kernel_zero)[0]))   


        #allocate memory for the output of the third kernel 
        gpu_image_Wxx_third_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
        gpu_image_Wyy_third_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
        gpu_image_Wxy_third_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)


        # Execute gaussian filter on all component matrices and calculate Harris Matrix
        program.filter_first_axis_second_pass(

                            queue, global_size, local_size, 
                            gpu_image_Wxx_derivative_out, 
                            gpu_image_Wyy_derivative_out, 
                            gpu_image_Wxy_derivative_out,
                            gpu_image_Wxx_third_out, 
                            gpu_image_Wyy_third_out, 
                            gpu_image_Wxy_third_out, 
                            local_memory_filter_Wxx, 
                            local_memory_filter_Wyy, 
                            local_memory_filter_Wxy,
                            halo, width, height, buf_width, buf_height, 
                            zero_kernel, local_buffer_zero_3

                            )


    ########################################### Fourth Kernel ##################################
    #                          This Kernel applies a gaussian to the product of the            #
    #                          parital derivatives in the x-direction (axis = 1)               #
    #                          and computes the final Harris Matrix for the output             #
    ########################################### Fourth Kernel ##################################

        #Allocate local memory buffer for fourth kernel
        local_memory_filter_Wxx_2 = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
        local_memory_filter_Wyy_2 = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
        local_memory_filter_Wxy_2 = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
        local_buffer_zero_4 = cl.LocalMemory(4 * (np.shape(filter_kernel_zero)[0]))   

        # Allocate memory to store output from fourth kernel, this is the final Harris Matrix
        gpu_image_filter_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)

        program.filter_second_axis_second_pass(

                            queue, global_size, local_size, 
                            gpu_image_Wxx_third_out, 
                            gpu_image_Wyy_third_out, 
                            gpu_image_Wxy_third_out, 
                            gpu_image_filter_out,
                            local_memory_filter_Wxx_2, 
                            local_memory_filter_Wyy_2, 
                            local_memory_filter_Wxy_2,
                            halo, width, height, buf_width, buf_height, 
                            zero_kernel, local_buffer_zero_4

                            )



        #Output the final Harris Matrix to the CPU
        cl.enqueue_copy(queue, Harris_Matrix, gpu_image_filter_out, is_blocking=False)
        points = get_harris_points(Harris_Matrix)
        end = time.time()

        #Store the time to run the entire openCL version
        output_times_openCL[i] = end - start

        #Store the time to run the entire serial version
        with Timer() as serial_time:
            harris = run_harris(host_image)
        output_times_serial[i] = serial_time.interval
 

    
    ######################################################################
    # test comparision for accuracy vs. harris.py Serial implementation by 
    # "Programming Computer Vision with Python"  by Jan Erik Solem
    ######################################################################

    print '-------------Check Plots: Saved to the Directory--------------------------'
    plot_harris_points(host_image, points, im_name = 'Harris openCL Image')
    response = compute_harris_response(host_image, sigma=1)
    serial_points = get_harris_points(response, min_dist=10, threshold=0.1)
    plot_harris_points(host_image, serial_points, im_name = 'Harris Serial Image')
    print '--------------------------------------------------------------------------'

    print '-------------Check For Correctness----------------------------------------'
    pt_x = np.random.randint(np.shape(host_image)[0])
    pt_y = np.random.randint(np.shape(host_image)[1])
    print 'openCL Harris Matrix Random Point Check:', Harris_Matrix[pt_x, pt_y]
    print 'Serial Baseline Harris Matrix Random Point Check:', response[pt_x, pt_y]
    print 'Number of openCL points:', np.shape(points)
    print 'Number of Serial Points:', np.shape(serial_points)
    print 'Are the two lists of corner points the same?', (np.array(serial_points) == np.array(points)).all()
    print '--------------------------------------------------------------------------'
    


    print '-------------Check Timing Comparision-------------------------------------'
    print 'Time to run openCL', output_times_openCL.mean()
    print 'Time to run Serial', output_times_serial.mean()
    print '--------------------------------------------------------------------------'












