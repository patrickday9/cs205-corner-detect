
// Zero and First Derivative of Gaussian filter along Y axis (verticle)
// Returns two arrays the size fo the image

__kernel void
gaussian_first_axis(
          __global __read_only float *in_values,
          __global __write_only float *out_zero,
          __global __write_only float *out_first,
          __local float *buffer,
          const int w, const int h,
          const int buf_w, const int buf_h,
          const int halo, 
          __global __read_only float *direct_der,
          __global __read_only float *zero_kernel,
          __local float *local_direct_der,
          __local float *local_zero_kernel

           )
{

    //////////////////////Define Variables///////////////////////

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    const int x = get_global_id(0); // values for the columns
    const int y = get_global_id(1); // values for the rows

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0); // these will be values for the columns
    const int ly = get_local_id(1); // these will be values for the rows

    // coordinates of the upper left corner of the buffer in image (global)
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    // this shifts the buffer reference to the middle of the buffer
    // where these pixels actualy exist in the buffer
    const int buf_x = lx + halo; // buffer position
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx; //get_local_size = 2

    // Load in referencing information for buffer looping
    const int max_x = buf_corner_x + idx_1D; // this is column index, add idx_1D
    const int new_h = h - 1; // height index
    const int new_w = w - 1; // width index

    //////////////////////load in local buffer for filter ///////////////////////
    if (idx_1D < 2 * halo + 1) 
    {
      local_direct_der[idx_1D] = direct_der[idx_1D];
      local_zero_kernel[idx_1D] = zero_kernel[idx_1D];
    }


    //////////////////////loop to build Buffer///////////////////////

    //Iterate down each colum, using a row iterator
        // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w) 
    {
	    for (int row = 0; row < buf_h; row++) 
	   	{

	      const int max_y = buf_corner_y + row; //stepping by rows adjust y

	      // Load the values into the buffer
	      // This is a read from global memory global read
	      // Each thread is loading values into the buffer down columns
	      buffer[row * buf_w + idx_1D] = in_values[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
	    }
	}


    //////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {
        // Y is in the axis = 1 direction
        const float yneighbor_0 = buffer[(buf_y - 4) * buf_w + (buf_x - 0)]; 
        const float yneighbor_1 = buffer[(buf_y - 3) * buf_w + (buf_x - 0)];
        const float yneighbor_2 = buffer[(buf_y - 2) * buf_w + (buf_x - 0)];
        const float yneighbor_3 = buffer[(buf_y - 1) * buf_w + (buf_x - 0)];
        const float yneighbor_5 = buffer[(buf_y + 1) * buf_w + (buf_x + 0)]; 
        const float yneighbor_6 = buffer[(buf_y + 2) * buf_w + (buf_x + 0)];
        const float yneighbor_7 = buffer[(buf_y + 3) * buf_w + (buf_x + 0)];
        const float yneighbor_8 = buffer[(buf_y + 4) * buf_w + (buf_x + 0)];
        const float ypixel = buffer[(buf_y + 0) * buf_w + (buf_x + 0)];

        //use a for loop to multiply by
        const float first = yneighbor_0 * local_direct_der[0] + yneighbor_1 * local_direct_der[1] + yneighbor_2 * 
        local_direct_der[2] + yneighbor_3 * local_direct_der[3] + yneighbor_5 * local_direct_der[5] + yneighbor_6 * local_direct_der[6] + 
        yneighbor_7 * local_direct_der[7] + yneighbor_8 * local_direct_der[8] + ypixel * local_direct_der[4];


        const float zero = yneighbor_0 * local_zero_kernel[0] + yneighbor_1 * local_zero_kernel[1] + yneighbor_2 * 
        local_zero_kernel[2] + yneighbor_3 * local_zero_kernel[3] + yneighbor_5 * local_zero_kernel[5] + yneighbor_6 * local_zero_kernel[6] + 
        yneighbor_7 * local_zero_kernel[7] + yneighbor_8 * local_zero_kernel[8] + ypixel * local_zero_kernel[4];



      	out_zero[y * w + x] = zero;
      	out_first[y * w + x] = first;

      }

}



// Zero and First Derivative of Gaussian filter along X axis (horizontal)
// Returns three arrays the size fo the image, where eached partial is squared


__kernel void
gaussian_second_axis(

          __global __read_only float *in_zero,
          __global __read_only float *in_first,
          __global __write_only float *out_Wxx,
          __global __write_only float *out_Wyy,
          __global __write_only float *out_Wxy,
          __local float *buffer_order0,
          __local float *buffer_order1,
          const int w, const int h,
          const int buf_w, const int buf_h,
          const int halo, 
          __global __read_only float *direct_der,
          __global __read_only float *zero_kernel,
          __local float *local_direct_der,
          __local float *local_zero_kernel

          )
{

    //////////////////////Define Variables///////////////////////

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    const int x = get_global_id(0); // values for the columns
    const int y = get_global_id(1); // values for the rows

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0); // these will be values for the columns
    const int ly = get_local_id(1); // these will be values for the rows

    // coordinates of the upper left corner of the buffer in image (global)
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    // this shifts the buffer reference to the middle of the buffer
    // where these pixels actualy exist in the buffer
    const int buf_x = lx + halo; // buffer position
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx; //get_local_size = 2

    // Load in referencing information for buffer looping
    const int max_x = buf_corner_x + idx_1D; // this is column index, add idx_1D
    const int new_h = h - 1; // height index
    const int new_w = w - 1; // width index

    //////////////////////load in local buffer for filter ///////////////////////
    if (idx_1D < 2 * halo + 1) 
    {
      local_direct_der[idx_1D] = direct_der[idx_1D];
      local_zero_kernel[idx_1D] = zero_kernel[idx_1D];
    }



    //////////////////////loop to build Buffer///////////////////////

    //Iterate down each colum, using a row iterator
        // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w) 
    {
      for (int row = 0; row < buf_h; row++) 
      {

        const int max_y = buf_corner_y + row; //stepping by rows adjust y

        // Load the values into the buffer
        // This is a read from global memory global read
        // Each thread is loading values into the buffer down columns
        buffer_order0[row * buf_w + idx_1D] = in_zero[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
        buffer_order1[row * buf_w + idx_1D] = in_first[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
      }
  }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    //barrier(CLK_LOCAL_MEM_FENCE);

    //////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {

        const float yneighbor_0_order0 = buffer_order0[(buf_y - 0) * buf_w + (buf_x - 4)]; 
        const float yneighbor_1_order0 = buffer_order0[(buf_y - 0) * buf_w + (buf_x - 3)];
        const float yneighbor_2_order0 = buffer_order0[(buf_y - 0) * buf_w + (buf_x - 2)];
        const float yneighbor_3_order0 = buffer_order0[(buf_y - 0) * buf_w + (buf_x - 1)];
        const float yneighbor_5_order0 = buffer_order0[(buf_y + 0) * buf_w + (buf_x + 1)]; 
        const float yneighbor_6_order0 = buffer_order0[(buf_y + 0) * buf_w + (buf_x + 2)];
        const float yneighbor_7_order0 = buffer_order0[(buf_y + 0) * buf_w + (buf_x + 3)];
        const float yneighbor_8_order0 = buffer_order0[(buf_y + 0) * buf_w + (buf_x + 4)];
        const float ypixel_order0 = buffer_order0[(buf_y + 0) * buf_w + (buf_x + 0)];

        const float yneighbor_0_order1 = buffer_order1[(buf_y - 0) * buf_w + (buf_x - 4)]; 
        const float yneighbor_1_order1 = buffer_order1[(buf_y - 0) * buf_w + (buf_x - 3)];
        const float yneighbor_2_order1 = buffer_order1[(buf_y - 0) * buf_w + (buf_x - 2)];
        const float yneighbor_3_order1 = buffer_order1[(buf_y - 0) * buf_w + (buf_x - 1)];
        const float yneighbor_5_order1 = buffer_order1[(buf_y + 0) * buf_w + (buf_x + 1)]; 
        const float yneighbor_6_order1 = buffer_order1[(buf_y + 0) * buf_w + (buf_x + 2)];
        const float yneighbor_7_order1 = buffer_order1[(buf_y + 0) * buf_w + (buf_x + 3)];
        const float yneighbor_8_order1 = buffer_order1[(buf_y + 0) * buf_w + (buf_x + 4)];
        const float ypixel_order1 = buffer_order1[(buf_y + 0) * buf_w + (buf_x + 0)];

        //use a for loop to multiply by
        const float Ix = yneighbor_0_order0 * local_direct_der[0] + yneighbor_1_order0 * local_direct_der[1] + yneighbor_2_order0 * 
        local_direct_der[2] + yneighbor_3_order0 * local_direct_der[3] + ypixel_order0 * local_direct_der[4] + yneighbor_5_order0 * 
        local_direct_der[5] + yneighbor_6_order0 * local_direct_der[6] + 
        yneighbor_7_order0 * local_direct_der[7] + yneighbor_8_order0 * local_direct_der[8];

        const float Iy = yneighbor_0_order1 * local_zero_kernel[0] + yneighbor_1_order1 * local_zero_kernel[1] + yneighbor_2_order1 * 
        local_zero_kernel[2] + yneighbor_3_order1 * local_zero_kernel[3] + ypixel_order1 * local_zero_kernel[4] + yneighbor_5_order1 * 
        local_zero_kernel[5] + yneighbor_6_order1 * local_zero_kernel[6] + 
        yneighbor_7_order1 * local_zero_kernel[7] + yneighbor_8_order1 * local_zero_kernel[8];


        out_Wxx[y * w + x] = Ix * Ix;
        out_Wyy[y * w + x] = Iy * Iy;
        out_Wxy[y * w + x] = Ix * Iy;

      }

}







// Performes a zero derivative gaussian on the Y axis (verticle)
// Returs 3 arrays the size of the image - all partials smoothed. 


__kernel void 
filter_first_axis_second_pass(

      __global __read_only float *in_Wxx,
      __global __read_only float *in_Wyy,
      __global __read_only float *in_Wxy, 
      __global __write_only float *out_Wxx,
      __global __write_only float *out_Wyy,
      __global __write_only float *out_Wxy,
      __local float *buffer_Wxx,
      __local float *buffer_Wyy,
      __local float *buffer_Wxy,
      const int halo,
      const int w, const int h,
      const int buf_w, const int buf_h,
      __global __read_only float *filter,
      __local float *local_zero_kernel

      )
{

   //////////////////////Define Variables///////////////////////

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0); // with in workgroup, so less than buffer
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image (global)
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    // this shifts the buffer reference to the middle of the buffer
    // where these pixels actualy exist in the buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx; //get_local_size = 8

    // Load in referencing information for buffer looping
    const int max_x = buf_corner_x + idx_1D; // this is column index, add idx_1D
    const int new_h = h - 1; // height index
    const int new_w = w - 1; // width index

    //////////////////////load in local buffer for filter ///////////////////////
    if (idx_1D < 2 * halo + 1) 
    {

      local_zero_kernel[idx_1D] = filter[idx_1D];

    }


    //////////////////////loop to build Buffer///////////////////////

    // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w) 
    {
        //Iterate down each colum, using a row iterator
        for (int row = 0; row < buf_h; row++) 
       {

          const int max_y = buf_corner_y + row; //stepping by rows adjust y

          // Load the values into the buffer
          // This is a read from global memory global read
          // Each thread is loading values into the buffer down columns
          buffer_Wxx[row * buf_w + idx_1D] = in_Wxx[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
          buffer_Wyy[row * buf_w + idx_1D] = in_Wyy[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
          buffer_Wxy[row * buf_w + idx_1D] = in_Wxy[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
        }

    }



//////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {

        const float neighbor_0_Wxx = buffer_Wxx[(buf_y - 4) * buf_w + (buf_x - 0)]; 
        const float neighbor_1_Wxx = buffer_Wxx[(buf_y - 3) * buf_w + (buf_x - 0)];
        const float neighbor_2_Wxx = buffer_Wxx[(buf_y - 2) * buf_w + (buf_x - 0)];
        const float neighbor_3_Wxx = buffer_Wxx[(buf_y - 1) * buf_w + (buf_x - 0)];
        const float neighbor_5_Wxx = buffer_Wxx[(buf_y + 1) * buf_w + (buf_x + 0)]; 
        const float neighbor_6_Wxx = buffer_Wxx[(buf_y + 2) * buf_w + (buf_x + 0)];
        const float neighbor_7_Wxx = buffer_Wxx[(buf_y + 3) * buf_w + (buf_x + 0)];
        const float neighbor_8_Wxx = buffer_Wxx[(buf_y + 4) * buf_w + (buf_x + 0)];
        const float pixel_Wxx = buffer_Wxx[(buf_y + 0) * buf_w + (buf_x + 0)];
        
        const float Wxx_filter = pixel_Wxx * (local_zero_kernel[4]) + neighbor_0_Wxx * (local_zero_kernel[0]) + neighbor_1_Wxx * (local_zero_kernel[1]) +
         neighbor_2_Wxx * (local_zero_kernel[2]) + neighbor_3_Wxx * (local_zero_kernel[3]) + neighbor_5_Wxx * (local_zero_kernel[5]) + 
         neighbor_6_Wxx * (local_zero_kernel[6]) + neighbor_7_Wxx * (local_zero_kernel[7]) + neighbor_8_Wxx * (local_zero_kernel[8]);
     

        const float neighbor_0_Wyy = buffer_Wyy[(buf_y - 4) * buf_w + (buf_x + 0)]; 
        const float neighbor_1_Wyy = buffer_Wyy[(buf_y - 3) * buf_w + (buf_x + 0)];
        const float neighbor_2_Wyy = buffer_Wyy[(buf_y - 2) * buf_w + (buf_x + 0)];
        const float neighbor_3_Wyy = buffer_Wyy[(buf_y - 1) * buf_w + (buf_x + 0)];
        const float neighbor_5_Wyy = buffer_Wyy[(buf_y + 1) * buf_w + (buf_x + 0)]; 
        const float neighbor_6_Wyy = buffer_Wyy[(buf_y + 2) * buf_w + (buf_x + 0)];
        const float neighbor_7_Wyy = buffer_Wyy[(buf_y + 3) * buf_w + (buf_x + 0)];
        const float neighbor_8_Wyy = buffer_Wyy[(buf_y + 4) * buf_w + (buf_x + 0)];
        const float pixel_Wyy = buffer_Wyy[(buf_y + 0) * buf_w + (buf_x + 0)];

        const float Wyy_filter = pixel_Wyy * (local_zero_kernel[4]) + neighbor_0_Wyy * (local_zero_kernel[0]) + neighbor_1_Wyy * (local_zero_kernel[1]) +
         neighbor_2_Wyy * (local_zero_kernel[2]) + neighbor_3_Wyy * (local_zero_kernel[3]) + neighbor_5_Wyy * (local_zero_kernel[5]) + 
         neighbor_6_Wyy * (local_zero_kernel[6]) + neighbor_7_Wyy * (local_zero_kernel[7]) + neighbor_8_Wyy * (local_zero_kernel[8]);
  

        const float neighbor_0_Wxy = buffer_Wxy[(buf_y - 4) * buf_w + (buf_x + 0)]; 
        const float neighbor_1_Wxy = buffer_Wxy[(buf_y - 3) * buf_w + (buf_x + 0)];
        const float neighbor_2_Wxy = buffer_Wxy[(buf_y - 2) * buf_w + (buf_x + 0)];
        const float neighbor_3_Wxy = buffer_Wxy[(buf_y - 1) * buf_w + (buf_x + 0)];
        const float neighbor_5_Wxy = buffer_Wxy[(buf_y + 1) * buf_w + (buf_x + 0)]; 
        const float neighbor_6_Wxy = buffer_Wxy[(buf_y + 2) * buf_w + (buf_x + 0)];
        const float neighbor_7_Wxy = buffer_Wxy[(buf_y + 3) * buf_w + (buf_x + 0)];
        const float neighbor_8_Wxy = buffer_Wxy[(buf_y + 4) * buf_w + (buf_x + 0)];
        const float pixel_Wxy = buffer_Wxy[(buf_y + 0) * buf_w + (buf_x + 0)];

        const float Wxy_filter = pixel_Wxy * (local_zero_kernel[4]) + neighbor_0_Wxy * (local_zero_kernel[0]) + neighbor_1_Wxy * (local_zero_kernel[1]) +
         neighbor_2_Wxy * (local_zero_kernel[2]) + neighbor_3_Wxy * (local_zero_kernel[3]) + neighbor_5_Wxy * (local_zero_kernel[5]) + 
         neighbor_6_Wxy * (local_zero_kernel[6]) + neighbor_7_Wxy * (local_zero_kernel[7]) + neighbor_8_Wxy * (local_zero_kernel[8]);
 

        out_Wxx[y * w + x] = Wxx_filter; 
        out_Wyy[y * w + x] = Wyy_filter;
        out_Wxy[y * w + x] = Wxy_filter;

      } 
  

}



// Performes a zero derivative gaussian on the X axis (verticle)
// Returs one array the size of the image
// Performs the harris eigen approximation and returns the full harris matrix 



__kernel void 
filter_second_axis_second_pass(

      __global __read_only float *in_Wxx,
      __global __read_only float *in_Wyy,
      __global __read_only float *in_Wxy, 
      __global __write_only float *out_Harris,
      __local float *buffer_Wxx,
      __local float *buffer_Wyy,
      __local float *buffer_Wxy,
      const int halo,
      const int w, const int h,
      const int buf_w, const int buf_h,
      __global __read_only float *filter,
      __local float *local_zero_kernel

      )
{

   //////////////////////Define Variables///////////////////////

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0); // with in workgroup, so less than buffer
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image (global)
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    // this shifts the buffer reference to the middle of the buffer
    // where these pixels actualy exist in the buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx; //get_local_size = 8

    // Load in referencing information for buffer looping
    const int max_x = buf_corner_x + idx_1D; // this is column index, add idx_1D
    const int new_h = h - 1; // height index
    const int new_w = w - 1; // width index

    //////////////////////load in local buffer for filter ///////////////////////

    if (idx_1D < 2 * halo + 1) 
    {

      local_zero_kernel[idx_1D] = filter[idx_1D];
 

    }

    //////////////////////loop to build Buffer///////////////////////

    // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w) 
    {
        //Iterate down each colum, using a row iterator
        for (int row = 0; row < buf_h; row++) 
       {

          const int max_y = buf_corner_y + row; //stepping by rows adjust y


          // Load the values into the buffer
          // This is a read from global memory global read
          // Each thread is loading values into the buffer down columns
          buffer_Wxx[row * buf_w + idx_1D] = in_Wxx[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
          buffer_Wyy[row * buf_w + idx_1D] = in_Wyy[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
          buffer_Wxy[row * buf_w + idx_1D] = in_Wxy[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
        }

    }



//////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {


        const float neighbor_0_Wxx = buffer_Wxx[(buf_y - 0) * buf_w + (buf_x - 4)]; 
        const float neighbor_1_Wxx = buffer_Wxx[(buf_y - 0) * buf_w + (buf_x - 3)];
        const float neighbor_2_Wxx = buffer_Wxx[(buf_y - 0) * buf_w + (buf_x - 2)];
        const float neighbor_3_Wxx = buffer_Wxx[(buf_y - 0) * buf_w + (buf_x - 1)];
        const float neighbor_5_Wxx = buffer_Wxx[(buf_y + 0) * buf_w + (buf_x + 1)]; 
        const float neighbor_6_Wxx = buffer_Wxx[(buf_y + 0) * buf_w + (buf_x + 2)];
        const float neighbor_7_Wxx = buffer_Wxx[(buf_y + 0) * buf_w + (buf_x + 3)];
        const float neighbor_8_Wxx = buffer_Wxx[(buf_y + 0) * buf_w + (buf_x + 4)];
        const float pixel_Wxx = buffer_Wxx[(buf_y + 0) * buf_w + (buf_x + 0)];

        const float Wxx_filter = pixel_Wxx * (local_zero_kernel[4]) + neighbor_0_Wxx * (local_zero_kernel[0]) + neighbor_1_Wxx * (local_zero_kernel[1]) +
         neighbor_2_Wxx * (local_zero_kernel[2]) + neighbor_3_Wxx * (local_zero_kernel[3]) + neighbor_5_Wxx * (local_zero_kernel[5]) + 
         neighbor_6_Wxx * (local_zero_kernel[6]) + neighbor_7_Wxx * (local_zero_kernel[7]) + neighbor_8_Wxx * (local_zero_kernel[8]);    

        const float neighbor_0_Wyy = buffer_Wyy[(buf_y - 0) * buf_w + (buf_x - 4)]; 
        const float neighbor_1_Wyy = buffer_Wyy[(buf_y - 0) * buf_w + (buf_x - 3)];
        const float neighbor_2_Wyy = buffer_Wyy[(buf_y - 0) * buf_w + (buf_x - 2)];
        const float neighbor_3_Wyy = buffer_Wyy[(buf_y - 0) * buf_w + (buf_x - 1)];
        const float neighbor_5_Wyy = buffer_Wyy[(buf_y + 0) * buf_w + (buf_x + 1)]; 
        const float neighbor_6_Wyy = buffer_Wyy[(buf_y + 0) * buf_w + (buf_x + 2)];
        const float neighbor_7_Wyy = buffer_Wyy[(buf_y + 0) * buf_w + (buf_x + 3)];
        const float neighbor_8_Wyy = buffer_Wyy[(buf_y + 0) * buf_w + (buf_x + 4)];
        const float pixel_Wyy = buffer_Wyy[(buf_y + 0) * buf_w + (buf_x + 0)];

        const float Wyy_filter = pixel_Wyy * (local_zero_kernel[4]) + neighbor_0_Wyy * (local_zero_kernel[0]) + neighbor_1_Wyy * (local_zero_kernel[1]) +
         neighbor_2_Wyy * (local_zero_kernel[2]) + neighbor_3_Wyy * (local_zero_kernel[3]) + neighbor_5_Wyy * (local_zero_kernel[5]) + 
         neighbor_6_Wyy * (local_zero_kernel[6]) + neighbor_7_Wyy * (local_zero_kernel[7]) + neighbor_8_Wyy * (local_zero_kernel[8]);
  

        const float neighbor_0_Wxy = buffer_Wxy[(buf_y - 0) * buf_w + (buf_x - 4)]; 
        const float neighbor_1_Wxy = buffer_Wxy[(buf_y - 0) * buf_w + (buf_x - 3)];
        const float neighbor_2_Wxy = buffer_Wxy[(buf_y - 0) * buf_w + (buf_x - 2)];
        const float neighbor_3_Wxy = buffer_Wxy[(buf_y - 0) * buf_w + (buf_x - 1)];
        const float neighbor_5_Wxy = buffer_Wxy[(buf_y + 0) * buf_w + (buf_x + 1)]; 
        const float neighbor_6_Wxy = buffer_Wxy[(buf_y + 0) * buf_w + (buf_x + 2)];
        const float neighbor_7_Wxy = buffer_Wxy[(buf_y + 0) * buf_w + (buf_x + 3)];
        const float neighbor_8_Wxy = buffer_Wxy[(buf_y + 0) * buf_w + (buf_x + 4)];
        const float pixel_Wxy = buffer_Wxy[(buf_y + 0) * buf_w + (buf_x + 0)];

        const float Wxy_filter = pixel_Wxy * (local_zero_kernel[4]) + neighbor_0_Wxy * (local_zero_kernel[0]) + neighbor_1_Wxy * (local_zero_kernel[1]) +
         neighbor_2_Wxy * (local_zero_kernel[2]) + neighbor_3_Wxy * (local_zero_kernel[3]) + neighbor_5_Wxy * (local_zero_kernel[5]) + 
         neighbor_6_Wxy * (local_zero_kernel[6]) + neighbor_7_Wxy * (local_zero_kernel[7]) + neighbor_8_Wxy * (local_zero_kernel[8]);
 

        const float Wdet = Wxx_filter * Wyy_filter - Wxy_filter * Wxy_filter;
        const float Wtr = Wxx_filter + Wyy_filter;
        const float ans = Wdet / Wtr;

        out_Harris[y * w + x] = ans; 

      } 
  

}




















