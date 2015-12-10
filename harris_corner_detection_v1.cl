

// Zero and First Derivative of Gaussian filter along Y axis (verticle)
// Returns two arrays the size fo the image

__kernel void
gaussian_first_axis(

           __global float *in_values,
           __global float *out_zero,
           __global float *out_first,
           int w, int h,
           __global float *direct_der,
           __global float *zero_kernel

           )
{

    //////////////////////Define Variables///////////////////////

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    int x = get_global_id(0); // values for the columns
    int y = get_global_id(1); // values for the rows


    //////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {
        // Y is in the axis = 1 direction
        float yneighbor_0 = in_values[((y - 4) * w) + (x - 0)]; 
        float yneighbor_1 = in_values[((y - 3) * w) + (x - 0)];
        float yneighbor_2 = in_values[((y - 2) * w) + (x - 0)];
        float yneighbor_3 = in_values[((y - 1) * w) + (x - 0)];
        float yneighbor_5 = in_values[((y + 1) * w) + (x + 0)]; 
        float yneighbor_6 = in_values[((y + 2) * w) + (x + 0)];
        float yneighbor_7 = in_values[((y + 3) * w) + (x + 0)];
        float yneighbor_8 = in_values[((y + 4) * w) + (x + 0)];
        float ypixel = in_values[((y + 0) * w) + (x + 0)];

        //use a for loop to multiply by
        float first = yneighbor_0 * direct_der[0] + yneighbor_1 * direct_der[1] + yneighbor_2 * 
        direct_der[2] + yneighbor_3 * direct_der[3] + yneighbor_5 * direct_der[5] + yneighbor_6 * direct_der[6] + 
        yneighbor_7 * direct_der[7] + yneighbor_8 * direct_der[8] + ypixel * direct_der[4];


        float zero = yneighbor_0 * zero_kernel[0] + yneighbor_1 * zero_kernel[1] + yneighbor_2 * 
        zero_kernel[2] + yneighbor_3 * zero_kernel[3] + yneighbor_5 * zero_kernel[5] + yneighbor_6 * zero_kernel[6] + 
        yneighbor_7 * zero_kernel[7] + yneighbor_8 * zero_kernel[8] + ypixel * zero_kernel[4];



      	out_zero[y * w + x] = zero;
      	out_first[y * w + x] = first;

      }

}




// Zero and First Derivative of Gaussian filter along X axis (horizontal)
// Returns three arrays the size fo the image, where eached partial is squared

__kernel void
gaussian_second_axis(
  
           __global float *in_zero,
           __global float *in_first,
           __global float *out_Wxx,
           __global float *out_Wyy,
           __global float *out_Wxy,
           int w, int h,
           __global float *direct_der,
           __global float *zero_kernel

           )
{

    //////////////////////Define Variables///////////////////////

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    int x = get_global_id(0); // values for the columns
    int y = get_global_id(1); // values for the rows



    //////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {

        float yneighbor_0_order0 = in_zero[(y - 0) * w + (x - 4)]; 
        float yneighbor_1_order0 = in_zero[(y - 0) * w + (x - 3)];
        float yneighbor_2_order0 = in_zero[(y - 0) * w + (x - 2)];
        float yneighbor_3_order0 = in_zero[(y - 0) * w + (x - 1)];
        float yneighbor_5_order0 = in_zero[(y + 0) * w + (x + 1)]; 
        float yneighbor_6_order0 = in_zero[(y + 0) * w + (x + 2)];
        float yneighbor_7_order0 = in_zero[(y + 0) * w + (x + 3)];
        float yneighbor_8_order0 = in_zero[(y + 0) * w + (x + 4)];
        float ypixel_order0 = in_zero[(y + 0) * w + (x + 0)];

        float yneighbor_0_order1 = in_first[(y - 0) * w + (x - 4)]; 
        float yneighbor_1_order1 = in_first[(y - 0) * w + (x - 3)];
        float yneighbor_2_order1 = in_first[(y - 0) * w + (x - 2)];
        float yneighbor_3_order1 = in_first[(y - 0) * w + (x - 1)];
        float yneighbor_5_order1 = in_first[(y + 0) * w + (x + 1)]; 
        float yneighbor_6_order1 = in_first[(y + 0) * w + (x + 2)];
        float yneighbor_7_order1 = in_first[(y + 0) * w + (x + 3)];
        float yneighbor_8_order1 = in_first[(y + 0) * w + (x + 4)];
        float ypixel_order1 = in_first[(y + 0) * w + (x + 0)];

        //use a for loop to multiply by
        float Ix = yneighbor_0_order0 * direct_der[0] + yneighbor_1_order0 * direct_der[1] + yneighbor_2_order0 * 
        direct_der[2] + yneighbor_3_order0 * direct_der[3] + ypixel_order0 * direct_der[4] + yneighbor_5_order0 * direct_der[5] + yneighbor_6_order0 * direct_der[6] + 
        yneighbor_7_order0 * direct_der[7] + yneighbor_8_order0 * direct_der[8];

        float Iy = yneighbor_0_order1 * zero_kernel[0] + yneighbor_1_order1 * zero_kernel[1] + yneighbor_2_order1 * 
        zero_kernel[2] + yneighbor_3_order1 * zero_kernel[3] + ypixel_order1 * zero_kernel[4] + yneighbor_5_order1 * zero_kernel[5] + yneighbor_6_order1 * zero_kernel[6] + 
        yneighbor_7_order1 * zero_kernel[7] + yneighbor_8_order1 * zero_kernel[8];

        //Return all paritals

        out_Wxx[y * w + x] = Ix * Ix;
        out_Wyy[y * w + x] = Iy * Iy;
        out_Wxy[y * w + x] = Ix * Iy;

      }

}



// Performes a zero derivative gaussian on the Y axis (verticle)
// Returs 3 arrays the size of the image - all partials smoothed. 

__kernel void 
filter_first_axis_second_pass(

      __global float *in_Wxx,
      __global float *in_Wyy,
      __global float *in_Wxy, 
      __global float *out_Wxx,
      __global float *out_Wyy,
      __global float *out_Wxy,

      int w, int h,
      __global __read_only float *filter

      )
{

   //////////////////////Define Variables///////////////////////

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    int x = get_global_id(0);
    int y = get_global_id(1);


//////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {


        float neighbor_0_Wxx = in_Wxx[((y - 4) * w) + (x - 0)]; 
        float neighbor_1_Wxx = in_Wxx[((y - 3) * w) + (x - 0)];
        float neighbor_2_Wxx = in_Wxx[((y - 2) * w) + (x - 0)];
        float neighbor_3_Wxx = in_Wxx[((y - 1) * w) + (x - 0)];
        float neighbor_5_Wxx = in_Wxx[((y + 1) * w) + (x - 0)]; 
        float neighbor_6_Wxx = in_Wxx[((y + 2) * w) + (x - 0)];
        float neighbor_7_Wxx = in_Wxx[((y + 3) * w) + (x - 0)];
        float neighbor_8_Wxx = in_Wxx[((y + 4) * w) + (x - 0)];
        float pixel_Wxx = in_Wxx[((y - 0) * w) + (x - 0)];
        //use a nested for loop
        float Wxx_filter = pixel_Wxx * (filter[4]) + neighbor_0_Wxx * (filter[0]) + neighbor_1_Wxx * (filter[1]) +
         neighbor_2_Wxx * (filter[2]) + neighbor_3_Wxx * (filter[3]) + neighbor_5_Wxx * (filter[5]) + 
         neighbor_6_Wxx * (filter[6]) + neighbor_7_Wxx * (filter[7]) + neighbor_8_Wxx * (filter[8]);
 
        //out_Wxx[y * w + x] = Wxx_filter;     

        float neighbor_0_Wyy = in_Wyy[((y - 4) * w) + (x - 0)]; 
        float neighbor_1_Wyy = in_Wyy[((y - 3) * w) + (x - 0)];
        float neighbor_2_Wyy = in_Wyy[((y - 2) * w) + (x - 0)];
        float neighbor_3_Wyy = in_Wyy[((y - 1) * w) + (x - 0)];
        float neighbor_5_Wyy = in_Wyy[((y + 1) * w) + (x - 0)]; 
        float neighbor_6_Wyy = in_Wyy[((y + 2) * w) + (x - 0)];
        float neighbor_7_Wyy = in_Wyy[((y + 3) * w) + (x - 0)];
        float neighbor_8_Wyy = in_Wyy[((y + 4) * w) + (x - 0)];
        float pixel_Wyy = in_Wyy[((y - 0) * w) + (x - 0)];

        float Wyy_filter = pixel_Wyy * (filter[4]) + neighbor_0_Wyy * (filter[0]) + neighbor_1_Wyy * (filter[1]) +
         neighbor_2_Wyy * (filter[2]) + neighbor_3_Wyy * (filter[3]) + neighbor_5_Wyy * (filter[5]) + 
         neighbor_6_Wyy * (filter[6]) + neighbor_7_Wyy * (filter[7]) + neighbor_8_Wyy * (filter[8]);
  

        float neighbor_0_Wxy = in_Wxy[((y - 4) * w) + (x - 0)]; 
        float neighbor_1_Wxy = in_Wxy[((y - 3) * w) + (x - 0)];
        float neighbor_2_Wxy = in_Wxy[((y - 2) * w) + (x - 0)];
        float neighbor_3_Wxy = in_Wxy[((y - 1) * w) + (x - 0)];
        float neighbor_5_Wxy = in_Wxy[((y + 1) * w) + (x - 0)]; 
        float neighbor_6_Wxy = in_Wxy[((y + 2) * w) + (x - 0)];
        float neighbor_7_Wxy = in_Wxy[((y + 3) * w) + (x - 0)];
        float neighbor_8_Wxy = in_Wxy[((y + 4) * w) + (x - 0)];
        float pixel_Wxy = in_Wxy[((y + 0) * w) + (x - 0)];

        float Wxy_filter = pixel_Wxy * (filter[4]) + neighbor_0_Wxy * (filter[0]) + neighbor_1_Wxy * (filter[1]) +
         neighbor_2_Wxy * (filter[2]) + neighbor_3_Wxy * (filter[3]) + neighbor_5_Wxy * (filter[5]) + 
         neighbor_6_Wxy * (filter[6]) + neighbor_7_Wxy * (filter[7]) + neighbor_8_Wxy * (filter[8]);
 

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

      __global float *in_Wxx,
      __global float *in_Wyy,
      __global float *in_Wxy, 
      __global float *out_Harris,
      int w, int h,
      __global __read_only float *filter

      )
{

   //////////////////////Define Variables///////////////////////

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    int x = get_global_id(0);
    int y = get_global_id(1);


//////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {


        float neighbor_0_Wxx = in_Wxx[(y - 0) * w + (x - 4)]; 
        float neighbor_1_Wxx = in_Wxx[(y - 0) * w + (x - 3)];
        float neighbor_2_Wxx = in_Wxx[(y - 0) * w + (x - 2)];
        float neighbor_3_Wxx = in_Wxx[(y - 0) * w + (x - 1)];
        float neighbor_5_Wxx = in_Wxx[(y + 0) * w + (x + 1)]; 
        float neighbor_6_Wxx = in_Wxx[(y + 0) * w + (x + 2)];
        float neighbor_7_Wxx = in_Wxx[(y + 0) * w + (x + 3)];
        float neighbor_8_Wxx = in_Wxx[(y + 0) * w + (x + 4)];
        float pixel_Wxx = in_Wxx[(y + 0) * w + (x + 0)];
        //use a nested for loop
        float Wxx_filter = pixel_Wxx * (filter[4]) + neighbor_0_Wxx * (filter[0]) + neighbor_1_Wxx * (filter[1]) +
         neighbor_2_Wxx * (filter[2]) + neighbor_3_Wxx * (filter[3]) + neighbor_5_Wxx * (filter[5]) + 
         neighbor_6_Wxx * (filter[6]) + neighbor_7_Wxx * (filter[7]) + neighbor_8_Wxx * (filter[8]);
 
        //out_Wxx[y * w + x] = Wxx_filter;     

        float neighbor_0_Wyy = in_Wyy[(y - 0) * w + (x - 4)]; 
        float neighbor_1_Wyy = in_Wyy[(y - 0) * w + (x - 3)];
        float neighbor_2_Wyy = in_Wyy[(y - 0) * w + (x - 2)];
        float neighbor_3_Wyy = in_Wyy[(y - 0) * w + (x - 1)];
        float neighbor_5_Wyy = in_Wyy[(y + 0) * w + (x + 1)]; 
        float neighbor_6_Wyy = in_Wyy[(y + 0) * w + (x + 2)];
        float neighbor_7_Wyy = in_Wyy[(y + 0) * w + (x + 3)];
        float neighbor_8_Wyy = in_Wyy[(y + 0) * w + (x + 4)];
        float pixel_Wyy = in_Wyy[(y + 0) * w + (x + 0)];

        float Wyy_filter = pixel_Wyy * (filter[4]) + neighbor_0_Wyy * (filter[0]) + neighbor_1_Wyy * (filter[1]) +
         neighbor_2_Wyy * (filter[2]) + neighbor_3_Wyy * (filter[3]) + neighbor_5_Wyy * (filter[5]) + 
         neighbor_6_Wyy * (filter[6]) + neighbor_7_Wyy * (filter[7]) + neighbor_8_Wyy * (filter[8]);
  

        float neighbor_0_Wxy = in_Wxy[(y - 0) * w + (x - 4)]; 
        float neighbor_1_Wxy = in_Wxy[(y - 0) * w + (x - 3)];
        float neighbor_2_Wxy = in_Wxy[(y - 0) * w + (x - 2)];
        float neighbor_3_Wxy = in_Wxy[(y - 0) * w + (x - 1)];
        float neighbor_5_Wxy = in_Wxy[(y + 0) * w + (x + 1)]; 
        float neighbor_6_Wxy = in_Wxy[(y + 0) * w + (x + 2)];
        float neighbor_7_Wxy = in_Wxy[(y + 0) * w + (x + 3)];
        float neighbor_8_Wxy = in_Wxy[(y + 0) * w + (x + 4)];
        float pixel_Wxy = in_Wxy[(y + 0) * w + (x + 0)];

        float Wxy_filter = pixel_Wxy * (filter[4]) + neighbor_0_Wxy * (filter[0]) + neighbor_1_Wxy * (filter[1]) +
         neighbor_2_Wxy * (filter[2]) + neighbor_3_Wxy * (filter[3]) + neighbor_5_Wxy * (filter[5]) + 
         neighbor_6_Wxy * (filter[6]) + neighbor_7_Wxy * (filter[7]) + neighbor_8_Wxy * (filter[8]);
 

        float Wdet = Wxx_filter * Wyy_filter - Wxy_filter * Wxy_filter;
        float Wtr = Wxx_filter + Wyy_filter;
        float ans = Wdet / Wtr;

        out_Harris[y * w + x] = ans; 

      } 
  

}




















