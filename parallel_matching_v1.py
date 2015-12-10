import pyopencl as cl
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from harris import *
import sys
import time

def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

def test_match(desc1, desc2, threshold=0.5):
    """ For each corner point descriptor in the first image,
    select its match to second image using
    normalized cross correlation. Return the Matching Matrix
    for comparison against the parallel version. """

    n = len(desc1[0])
    d = -np.ones((len(desc1),len(desc2))) # pair-wise distances
    
    start = time.time()

    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j]) 
            ncc_value = sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value

    ndx = np.argsort(-d)
    
    end = time.time()
    print "Serial Execution Time Matching", end - start
    print

    matchscores = ndx[:,0] 
    
    return matchscores, d

def parallel_twosided(matches_12, matches_21, threshold=0.5):
    """ Two-sided symmetric version of match(). Used with parallel
    match scores. """
    ndx_12 = np.where(matches_12 >= 0)[0]
    
    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1 
            
    return matches_12

def parallel_matching(harris_im1, harris_im2, filtered_coords1, filtered_coords2, runs=1, 
                        verbose=True, opencl_file='match_preprocess_online_v4.cl', run_serial=False):

    # List our platforms
    platforms = cl.get_platforms()

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices)

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)

    ### Run Specific OpenCL File ###     
    program = cl.Program(context, open(opencl_file).read()).build(options='')

    if verbose == False:
        print 'The platforms detected are:'
        print '---------------------------'
        for platform in platforms:
            print platform.name, platform.vendor, 'version:', platform.version

        # List devices in each platform
        for platform in platforms:
            print 'The devices detected on platform', platform.name, 'are:'
            print '---------------------------'
            for device in platform.get_devices():
                print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
                print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
                print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
                print 'Maximum work group size', device.max_work_group_size
                print '---------------------------'
        
        print 'This context is associated with ', len(context.devices), 'devices'
        print 'The queue is using the device:', queue.device.name

    #########################################
    ### Harris Points and Variable Setup  ###
    #########################################

    # Find the Harris Corners, descriptors, and Haris Matrix
    # for Image 1 with a window width of 5
    wid = np.int32(5)
    d1 = get_descriptors(harris_im1, filtered_coords1, wid)
    
    # Convert all values to float/int 32
    har32_im1 = harris_im1.astype(np.float32)
    harris_corner1 = np.array(filtered_coords1).astype(np.int32)
    num_corners1 = np.int32(harris_corner1.shape[0])
    d1_32 = np.array(d1).astype(np.float32)

    # Same thing for Image 2 and convert to 32 bit
    d2 = get_descriptors(harris_im2, filtered_coords2, wid)

    # Convert all values to 32 bit
    har32_im2 = harris_im2.astype(np.float32)
    harris_corner2 = np.array(filtered_coords2).astype(np.int32)
    num_corners2 = np.int32(harris_corner2.shape[0])
    d2_32 = np.array(d2).astype(np.float32)

    #########################################
    ### Start Parallel                    ###
    ### For Image 1 Comparsion to Image 2 ###
    #########################################
    # Allocate GPU Memory for Image 1 Harris Corner List
    gpu_im1 = cl.Buffer(context, cl.mem_flags.READ_WRITE, har32_im1.size * 4)
    gpu_im2 = cl.Buffer(context, cl.mem_flags.READ_WRITE, har32_im2.size * 4)

    # Copy both sets of corners to the GPU
    cl.enqueue_copy(queue, gpu_im1, har32_im1, is_blocking=False)
    cl.enqueue_copy(queue, gpu_im2, har32_im2, is_blocking=False)

    # Allocate Local and Global Memory Buffer Size
    # Global Size is the amount of corners in Image1
    local_size = (1, 1)
    global_size = (1, harris_corner1.shape[0])
    
    # Set up Constant Parameters
    # Buffer Size is Descriptor Neighbor size 11x11 
    # Height and Width is the Shape of the Image 1
    buf_height = np.int32(11)
    buf_width = np.int32(11)
    
    height1 = np.int32(harris_im1.shape[0])
    width1 = np.int32(harris_im1.shape[1])
    
    threshold = np.float32(0.5)

    # Allocate local buffer for storing image neighbors
    local_memory1 = cl.LocalMemory(4 * (buf_width * buf_height))
    local_memory2 = cl.LocalMemory(4 * (buf_width * buf_height))

    if verbose == False:
        print "GPU Global Size:", global_size
        print "GPU Local Size:", local_size
        print "Harris returns corner points like this:", filtered_coords1[0], filtered_coords2[0]

    print "==========================================================="
    print 'Starting Parallel Matching'
    print "Using matching:", opencl_file, '\n'
    print "Shape of Image 1:", np.shape(harris_im1)
    print "Number of Corners found in Image 1:", harris_corner1.shape[0]
    print 
    
    print "Shape of Image 2:", np.shape(harris_im2)
    print "Number of Corners found in Image 2:", harris_corner2.shape[0]

    # Array for Corner 1 Match Scores
    py_match1 = -np.ones((len(d1),len(d2))).astype(np.float32)
    gpu_match1 = cl.Buffer(context, cl.mem_flags.READ_WRITE, py_match1.size * 4)
    cl.enqueue_copy(queue, gpu_match1, py_match1, is_blocking=False)
    
    # Total Harris Corners Matrix for Image 1 and 2
    gpu_harris1 = cl.Buffer(context, cl.mem_flags.READ_WRITE, harris_corner1.size*4)
    gpu_harris2 = cl.Buffer(context, cl.mem_flags.READ_WRITE, harris_corner2.size*4)
    cl.enqueue_copy(queue, gpu_harris1, harris_corner1, is_blocking=False)
    cl.enqueue_copy(queue, gpu_harris2, harris_corner2, is_blocking=False)

    # Run OpenCL Code Multiple times for average runtime
    avg_time = []
    runs = int(runs)
    for avg in xrange(1, runs+1):
        # Start the timer
        p_start1 = time.time()

        # Call OpenCL Function for the Naive Two Pass Implementation
        # Compare Corners Mapped By Both Images and return to GPU_Match
        event_compare1 = program.corner_match(queue, global_size, local_size, 
                            gpu_im1, gpu_im2, gpu_harris1, gpu_harris2, 
                            local_memory1, local_memory2, buf_width, buf_height, 
                            width1, height1, np.int32(wid), threshold, 
                            num_corners2, gpu_match1)

        # Copy Match Matrix back from GPU to CPU
        cl.enqueue_copy(queue, py_match1, gpu_match1, is_blocking=False)
        
        # Complete the serial argsort
        ndx1 = np.argsort(-py_match1)

        # End Timer, Parallel Code is Complete
        p_end1 = time.time()
        avg_time.append(p_end1 - p_start1)
    
    # Verify that all Parallel Corners Match Serial
    print "\nParallel Execution Time matching I1 to I2:", np.sum(avg_time)/runs
    if runs > 1: print "Averaged over", runs, "runs"

    # Save the first column of matches
    parallel_match1 = ndx1[:,0] 
    
    # Run Serial Matching for Timing Comparsion
    if run_serial == True:
        # Run the serial version of match an return match matrix
        serial_match1, cpu_d = test_match(d1, d2)

    #########################################
    ### Start Parallel                    ###
    ### For Image 2 Comparsion to Image 1 ###
    #########################################

    # Constants
    height2 = np.int32(harris_im2.shape[0])
    width2 = np.int32(harris_im2.shape[1])
    
    # Array for Corner Match Scores
    py_match2 = -np.ones((len(d2),len(d1))).astype(np.float32)
    gpu_match2 = cl.Buffer(context, cl.mem_flags.READ_WRITE, py_match2.size * 4)
    cl.enqueue_copy(queue, gpu_match2, py_match2, is_blocking=False)
    
    # Run the parallel matching mulitple times for average run time
    avg_time2 = []
    for avg in xrange(1, runs+1):
        # Start the timer
        start2 = time.time()

        # Call OpenCL function 
        # Compare Corners from Image 2 to Image 1
        event_compare2 = program.corner_match(queue, global_size, local_size, 
                            gpu_im2, gpu_im1, gpu_harris2, gpu_harris1,
                            local_memory1, local_memory2,  buf_width, buf_height, 
                            width2, height2, np.int32(wid), threshold, 
                            num_corners1, gpu_match2)

        # Copy back from GPU to CPU and end timer
        cl.enqueue_copy(queue, py_match2, gpu_match2, is_blocking=False)
        
        # Complete the argsort
        ndx2 = np.argsort(-py_match2)
    
        end2 = time.time()
        avg_time2.append(end2 - start2)
    
    print "Parallel Execution Time matching I2 to I1:", np.sum(avg_time2)/runs
    if runs > 1: print "Averaged over", runs, "runs"

    # Save the first column for matches
    parallel_match2 = ndx2[:,0] 
    
    if run_serial == True:    
        # Run the serial version of match and return match matrix
        serial_match2, cpu_d2 = test_match(d2, d1)

    # Run the Parallel and Serial Version then Compare   
    matches_parallel = parallel_twosided(parallel_match1, parallel_match2)    
    
    print "==========================================================="

    return matches_parallel


def match_plot(im1, im2, locs1, locs2, matchscores, show_below=True):
    im3 = appendimages(im1,im2) 
    if show_below:
        im3 = np.vstack((im3,im3))
    
    plt.figure(figsize=[12,8])
    
    plt.gray() 
    plt.imshow(im3, aspect='auto')
    
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plt.plot([locs1[i][1],locs2[m][1]+cols1], [locs1[i][0],locs2[m][0]], 'c')
            plt.axis('off')
    
    plt.show()

if __name__ == '__main__':

    ''' Import both images '''
    if len(sys.argv) > 3: 
        image1 = sys.argv[1]
        image2 = sys.argv[2]
        num_runs = sys.argv[3]
        np_im1 = np.array(Image.open(image1).convert('L')).astype(np.float32)[::1, ::1].copy()
        np_im2 = np.array(Image.open(image2).convert('L')).astype(np.float32)[::1, ::1].copy()
        print "==========================================================="
        print "Running Harris Corner detection for", image1, "and", image2


    else:
        print 'No image specified'
        print 'Run as follows: parallel_mapping_final.py <image1.png> <image2.jpg> <num_runs>\n' 
    
    # Stand Alone Version Runs Harris
    wid = 1
    harris_im1 = compute_harris_response(np_im1, wid) 
    filtered_coords1 = get_harris_points(harris_im1)
    harris_im2 = compute_harris_response(np_im2, wid) 
    filtered_coords2 = get_harris_points(harris_im2)

    # Execute the Parallel Matching Algorthim
    parallel_matches = parallel_matching(harris_im1, harris_im2, filtered_coords1, filtered_coords2, 
        opencl_file='match_preprocess_online_v1.cl', runs=num_runs, run_serial=True)
    
    # Plot results
    match_plot(np_im1, np_im2, filtered_coords1, filtered_coords2, parallel_matches, show_below=False)


