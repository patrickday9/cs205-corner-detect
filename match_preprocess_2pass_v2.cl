
// Parallel Mapping Comparsion
__kernel void
corner_match(__global __read_only float *image1,
           __global __read_only float *image2,
           __global __read_only int *harris1,
           __global __read_only int *harris2,
           __local float *buf_im1,
           __local float *buf_im2,
           int buf_w, int buf_h,
           int w, int h, int neighbor_wid, 
           float threshold, int num_cor2,
           __global __write_only float *d_matrix)
{

  //////////////////////Define Variables///////////////////////

  // halo is the additional number of cells in one direction

  // Global position of output pixel
  const int x = get_global_id(0); // values for the columns
  const int y = get_global_id(1); // values for the rows

  // Local position relative to (0, 0) in workgroup
  const int lx = get_local_id(0); // these will be values for the columns
  const int ly = get_local_id(1); // these will be values for the rows

    
  // Get the harris corner values based on global id
  int cor1_row = harris1[y*2 + x];   // [->y, x]
  int cor1_col = harris1[y*2 + x+1]; // [y, ->x] 
  
  // Initialze sum1 and mean1 for each thread
  float sum1 = 0;

  // Get all neighbor pixels
  for (int row = -neighbor_wid; row < neighbor_wid+1; row++){
    for (int col = -neighbor_wid; col < neighbor_wid+1; col++){
     sum1 += image1[(row + cor1_row) * w + (col + cor1_col)];
    }
  }

  // Calculate Mean and Sum
  float mean1 = sum1 / float(buf_w * buf_h);

  ///////////////////////
  // Begin all Corners //
  ///////////////////////
  for (int y_cor2 = 0; y_cor2 < num_cor2; y_cor2++) {  

    // Get Image 2 corners
    int cor2_row = harris2[y_cor2 * 2 + 0];
    int cor2_col = harris2[y_cor2 * 2 + 1]; 
    
    // Reitialize the sum and mean after each iteration
    float sum2 = 0;
    
    for (int row = -neighbor_wid; row < neighbor_wid+1; row++) {
      for (int col = -neighbor_wid; col < neighbor_wid+1; col++) {
        sum2 += image2[(row+cor2_row) * w + (col+cor2_col)];
      }
    }

    // Calculate Mean
    float mean2 = sum2 / float(buf_w * buf_h);        

    // Calculate Sum of Squared of each item
    float sum_of_sqrs1 = 0;
    float sum_of_sqrs2 = 0;

    // Calculate Sum of Squares Difference
    for (int row = -neighbor_wid; row < neighbor_wid+1; row++){
      for (int col = -neighbor_wid; col < neighbor_wid+1; col++){
        sum_of_sqrs1 += ( (image1[(row+cor1_row) * w + (col+cor1_col)] - mean1) 
                        * (image1[(row+cor1_row) * w + (col+cor1_col)] - mean1) );
        sum_of_sqrs2 += ( (image2[(row+cor2_row) * w + (col+cor2_col)] - mean2) 
                        * (image2[(row+cor2_row) * w + (col+cor2_col)] - mean2) );
      }
    }

    // Calculate Standard Deviation and Store to output array
    float std1 = sqrt( sum_of_sqrs1 / (buf_w*buf_h) );
    float std2 = sqrt( sum_of_sqrs2 / (buf_w*buf_h) );

    float ncc_sum = 0;
    // Calculate Normalized Cross Correlation
    for (int row = -neighbor_wid; row < neighbor_wid+1; row++){
      for (int col = -neighbor_wid; col < neighbor_wid+1; col++){
        
        // Calculate point by point matrix multiplication
        ncc_sum += ( (image1[(row+cor1_row) * w + (col+cor1_col)] - mean1) / std1)
                 * ( (image2[(row+cor2_row) * w + (col+cor2_col)] - mean2) / std2);
      }
    }

    // If NCC is above threshold, save to match matrix
    float ncc_value = ncc_sum/((buf_w*buf_h)-1);
    if (ncc_value > threshold){
      d_matrix[y * num_cor2 + y_cor2] = ncc_value;
    }

  } // End of Sum loops           
}
