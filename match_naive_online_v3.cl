
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

  // Begin all Corners //
  for (int y_cor2 = 0; y_cor2 < num_cor2; y_cor2++) {  

    // Get Image 2 corners
    int cor2_row = harris2[y_cor2 * 2 + 0];
    int cor2_col = harris2[y_cor2 * 2 + 1]; 

    // Declare Variables needed for Online Algo    
    float temp1 = 0;
    float temp2 = 0;
    float n1 = 0;
    float n2 = 0;
    float delta1 = 0;
    float delta2 = 0;
    float mean1 = 0;
    float mean2 = 0;
    float M1 = 0;
    float M2 = 0;
    
    // Compuit  
    for (int row = -neighbor_wid; row < neighbor_wid+1; row++){
      for (int col = -neighbor_wid; col < neighbor_wid+1; col++){
        
        // Store Value in temp, called twice below
        temp1 = image1[(row+cor1_row) * w + (col+cor1_col)];
        temp2 = image2[(row+cor2_row) * w + (col+cor2_col)];
        
        // Total Element Counter
        n1 += 1;
        n2 += 1;

        // Diff between current pixel and mean
        delta1 = temp1 - mean1;
        delta2 = temp2 - mean2;
        
        // Divide by total number of elements so far
        mean1 += delta1 / n1;
        mean2 += delta2 / n2;

        // M gives SS of Differences
        M1 += delta1 * (temp1 - mean1);
        M2 += delta2 * (temp2 - mean2);

      }
    }

    // Calculate Standard Deviation and Store to output array
    float std1 = sqrt(M1/(n1));
    float std2 = sqrt(M2/(n2));

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
