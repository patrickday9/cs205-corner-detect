
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
    const int buf_corner_x = x - lx;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    // this shifts the buffer reference to the middle of the buffer
    // where these pixels actualy exist in the buffer
    const int buf_x = lx; // buffer position
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx; //get_local_size = 2

    // 1D length of the entire kernel (9 for sigma of 1)
    const int window_length = 2 * halo + 1;
    const int neighbors = 2 * halo - 1; //number of neighbors less one for the bank conflict

    // Load in referencing information for buffer looping
    const int max_x = buf_corner_x + idx_1D; // this is column index, add idx_1D
    const int new_h = h - 1; // height index
    const int new_w = w - 1; // width index


    //////////////////////load in local buffer for filter ///////////////////////
    if (idx_1D < window_length + 1) 
    {
      local_direct_der[idx_1D] = direct_der[idx_1D];
      local_zero_kernel[idx_1D] = zero_kernel[idx_1D];
    }


    //////////////////////loop to build Buffer///////////////////////

    //Iterate down each colum, using a row iterator
        // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w - neighbors) 
    {
      for (int row = 0; row < buf_h; row++) 
      {

        const int max_y = buf_corner_y + row; //stepping by rows adjust y

        // Load the values into the buffer
        // This is a read from global memory global read
        // Each thread is loading values into the buffer down columns
        buffer[row * (buf_w - neighbors) + idx_1D] = in_values[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
      }
  }


    //////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {

        float zero = 0;
        float first = 0;

         for (int psn = 0; psn < window_length; psn++) 
        {
          zero += buffer[(buf_y + (psn - halo)) * (buf_w - neighbors) + (buf_x - 0)] * local_zero_kernel[psn];
          first += buffer[(buf_y + (psn - halo)) * (buf_w - neighbors) + (buf_x - 0)] * local_direct_der[psn];


        }

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
    const int buf_corner_y = y - ly;

    // coordinates of our pixel in the local buffer
    // this shifts the buffer reference to the middle of the buffer
    // where these pixels actualy exist in the buffer
    const int buf_x = lx + halo; // buffer position
    const int buf_y = ly;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx; //get_local_size = 2

    // 1D length of the entire kernel (9 for sigma of 1)
    const int window_length = 2 * halo + 1;
    const int neighbors = 2 * halo; //number of neighbors 

    // Load in referencing information for buffer looping
    const int max_x = buf_corner_x + idx_1D; // this is column index, add idx_1D
    const int new_h = h - 1; // height index
    const int new_w = w - 1; // width index

    //////////////////////load in local buffer for filter ///////////////////////
    if (idx_1D < window_length + 1) 
    {
      local_direct_der[idx_1D] = direct_der[idx_1D];
      local_zero_kernel[idx_1D] = zero_kernel[idx_1D];
    }


    //////////////////////loop to build Buffer///////////////////////

    //Iterate down each colum, using a row iterator
        // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w + 1) 
    {
      for (int row = 0; row < buf_h - neighbors; row++) 
      {

        const int max_y = buf_corner_y + row; //stepping by rows adjust y

        // Load the values into the buffer
        // This is a read from global memory global read
        // Each thread is loading values into the buffer down columns
        buffer_order0[row * buf_w + idx_1D] = in_zero[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
        buffer_order1[row * buf_w + idx_1D] = in_first[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
      }
  }


    //////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {

        float Iy = 0;
        float Ix = 0;

         for (int psn = 0; psn < window_length; psn++) 
        {
          Iy += buffer_order1[(buf_y + 0) * buf_w + (buf_x + (psn - halo))] * local_zero_kernel[psn];
          Ix += buffer_order0[(buf_y + 0) * buf_w + (buf_x + (psn - halo))] * local_direct_der[psn];


        }

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
    const int buf_corner_x = x - lx;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    // this shifts the buffer reference to the middle of the buffer
    // where these pixels actualy exist in the buffer
    const int buf_x = lx;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx; //get_local_size = 8

    // 1D length of the entire kernel (9 for sigma of 1)
    const int window_length = 2 * halo + 1;
    const int neighbors = 2 * halo - 1; //number of neighbos less one for the bank conflict

    // Load in referencing information for buffer looping
    const int max_x = buf_corner_x + idx_1D; // this is column index, add idx_1D
    const int new_h = h - 1; // height index
    const int new_w = w - 1; // width index

    //////////////////////load in local buffer for filter ///////////////////////
    if (idx_1D < window_length + 1) 
    {

      local_zero_kernel[idx_1D] = filter[idx_1D];

    }


    //////////////////////loop to build Buffer///////////////////////

    // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w - neighbors) 
    {
        //Iterate down each colum, using a row iterator
        for (int row = 0; row < buf_h; row++) 
       {

          const int max_y = buf_corner_y + row; //stepping by rows adjust y

          // Load the values into the buffer
          // This is a read from global memory global read
          // Each thread is loading values into the buffer down columns
          buffer_Wxx[row * (buf_w - neighbors) + idx_1D] = in_Wxx[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
          buffer_Wyy[row * (buf_w - neighbors) + idx_1D] = in_Wyy[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
          buffer_Wxy[row * (buf_w - neighbors) + idx_1D] = in_Wxy[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
        }

    }


//////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {

        float Wxx_filter = 0;
        float Wyy_filter = 0;
        float Wxy_filter = 0;

         for (int psn = 0; psn < window_length; psn++) 
        {
          Wxx_filter += buffer_Wxx[(buf_y + (psn - halo)) * (buf_w - neighbors) + (buf_x + 0)] * local_zero_kernel[psn];
          Wyy_filter += buffer_Wyy[(buf_y + (psn - halo)) * (buf_w - neighbors) + (buf_x + 0)] * local_zero_kernel[psn];
          Wxy_filter += buffer_Wxy[(buf_y + (psn - halo)) * (buf_w - neighbors) + (buf_x + 0)] * local_zero_kernel[psn];
        }

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
    const int buf_corner_y = y - ly;

    // coordinates of our pixel in the local buffer
    // this shifts the buffer reference to the middle of the buffer
    // where these pixels actualy exist in the buffer
    const int buf_x = lx + halo;
    const int buf_y = ly;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx; //get_local_size = 8

    // 1D length of the entire kernel (9 for sigma of 1)
    const int window_length = 2 * halo + 1; 
    const int neighbors = 2 * halo; //number of neighbors 

    // Load in referencing information for buffer looping
    const int max_x = buf_corner_x + idx_1D; // this is column index, add idx_1D
    const int new_h = h - 1; // height index
    const int new_w = w - 1; // width index

    //////////////////////load in local buffer for filter ///////////////////////

    if (idx_1D < window_length + 1) 
    {

      local_zero_kernel[idx_1D] = filter[idx_1D];
 

    }


    //////////////////////loop to build Buffer///////////////////////

    // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w + 1) 
    {
        //Iterate down each colum, using a row iterator
        for (int row = 0; row < buf_h - neighbors; row++) 
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

        float Wxx_filter = 0;
        float Wyy_filter = 0;
        float Wxy_filter = 0;

         for (int psn = 0; psn < window_length; psn++) 
        {
          Wxx_filter += buffer_Wxx[(buf_y + 0) * buf_w + (buf_x + (psn - halo))] * local_zero_kernel[psn];
          Wyy_filter += buffer_Wyy[(buf_y + + 0) * buf_w + (buf_x + (psn - halo))] * local_zero_kernel[psn];
          Wxy_filter += buffer_Wxy[(buf_y + 0) * buf_w + (buf_x + (psn - halo))] * local_zero_kernel[psn];
        }

        const float Wdet = Wxx_filter * Wyy_filter - Wxy_filter * Wxy_filter;
        const float Wtr = Wxx_filter + Wyy_filter;
        const float ans = Wdet / Wtr;

        out_Harris[y * w + x] = ans; 

      } 
  

}




















