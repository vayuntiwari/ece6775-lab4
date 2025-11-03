//===========================================================================
// layer.h
//===========================================================================
// @brief: This header file defines the interface for the core functions.

#ifndef LAYER_H
#define LAYER_H

#include "model.h"
#include "typedefs.h"

//----------------------------------------------------------
// Padding
//----------------------------------------------------------
// @param[in] : input - input fmaps
//              M - number of input fmaps
//              I - width of input fmaps
// @param[out] : output - output fmaps
template <int M, int I>
void pad(bit input[M][I][I], bit output[M][I + F_PAD][I + F_PAD]) {

  for (int m = 0; m < M; m++) {
      for (int y = 0; y < I; y++) {
      #pragma HLS pipeline
    for (int x = 0; x < I; x++) {
        output[m][y + F_PAD / 2][x + F_PAD / 2] = input[m][y][x];
      }
    }
  }
}

//----------------------------------------------------------
// Initialize Padded Memory with Constant
//----------------------------------------------------------
// @param[in] : input - input fmaps to be initialized
// @param[out] : output - output fmaps
template <int M, int I, int C>
void initialize_padded_memory(bit input[M][I][I]) {
  for (int m = 0; m < M; m++) {
    for (int x = 0; x < I; x++) {
      #pragma HLS pipeline
      for (int y = 0; y < I; y++) {
        input[m][x][y] = C;
      }
    }
  }
}

//----------------------------------------------------------
// Perform Convolution Layer
//----------------------------------------------------------
// @param[in] : input - input fmaps
//              threshold - threshold for batchnorm operation
//              M - number of input fmaps
//              N - number of output fmaps
//              I - width of input fmaps
//              weight - layer weights
// @param[out] : output - output fmaps
template <int M, int N, int I>
void conv(bit input[M][I][I], bit output[N][I - F + 1][I - F + 1],
          const bit8_t threshold[N], const bit weight[M][N][F][F]) {
  const int num_accum = F * F * M;

  // window holds current F x F patch for each channel m
  bit window[M][F][F];

  // linebuffer holds the F-1 previous rows (relative to current y)
  // indexed as linebuffer[m][r][x] where r in [0 .. F-2] corresponds to row (y + r)

  #pragma HLS array_reshape variable=window complete dim=1

  // Initialize linebuffer for y = 0 with rows [0 .. F-2]

  // Main scan: for each output row y, slide across columns x
  for (int n = 0; n < N; ++n) {
    bit linebuffer[M][F - 1][I];
    #pragma HLS array_reshape variable=linebuffer complete dim=1
    #pragma HLS array_partition variable=linebuffer complete dim=2
    #pragma HLS array_partition variable=linebuffer complete dim=3
    for (int m = 0; m < M; ++m) {
      #pragma HLS pipeline
      for (int r = 0; r < F - 1; ++r) {
        for (int x = 0; x < I; ++x) {
          linebuffer[m][r][x] = input[m][r][x];
        }
      }
    }
    for (int y = 0; y < I - F + 1; ++y) {
      #pragma HLS pipeline

      // For the current y, the "bottom" row index for the window is (y + F - 1)
      const int bottom_row = y + F - 1;

      // For each x across the row
      for (int x = 0; x < I - F + 1; ++x) {
        // If x == 0, build the full window from linebuffer (rows y..y+F-2) and input bottom_row
        if (x == 0) {
          for (int m = 0; m < M; ++m) {
            // rows r = 0 .. F-2 come from linebuffer[m][r][c]
            for (int r = 0; r < F - 1; ++r) {
              for (int c = 0; c < F; ++c) {
                // column c within window maps to absolute column c (since x==0)
                window[m][r][c] = linebuffer[m][r][c];
              }
            }
            // bottom row r = F-1 comes from input[m][bottom_row][c]
            for (int c = 0; c < F; ++c) {
              window[m][F - 1][c] = input[m][bottom_row][c];
            }
          }
        } else {
          // shift window left by one column, then load the new right column from:
          // - linebuffer for rows 0..F-2 at absolute column (x + F - 1)
          // - input bottom_row at column (x + F - 1) for row F-1
          const int new_col = x + F - 1;
          for (int m = 0; m < M; ++m) {
            // shift each row's columns left
            for (int r = 0; r < F; ++r) {
              for (int c = 0; c < F - 1; ++c) {
                window[m][r][c] = window[m][r][c + 1];
              }
            }
            // load new rightmost column
            for (int r = 0; r < F - 1; ++r) {
              window[m][r][F - 1] = linebuffer[m][r][new_col];
            }
            window[m][F - 1][F - 1] = input[m][bottom_row][new_col];
          }
        }

        // Compute accumulation from the window (consistent with debug checks)
        bit16_t accum = 0;
        for (int r = 0; r < F; ++r) {
          for (int c = 0; c < F; ++c) {
            for (int m = 0; m < M; ++m) {
              accum += (window[m][r][c] == weight[m][n][r][c]);
            }
          }
        }
        accum = (accum << 1) - num_accum;
        output[n][y][x] = accum > threshold[n] ? 1 : 0;

      } // x

      // After finishing the row of x, update linebuffer so that it corresponds to next y:
      // shift rows up (r := r+1), and insert new last row = input[*, y+F-1, *]
      // After this update, linebuffer will represent rows [y+1 .. y+F-1]
      for (int m = 0; m < M; ++m) {
        for (int x = 0; x < I; ++x) {
          for (int r = 0; r < F - 2; ++r) {
          // shift up rows 0..F-3 <- 1..F-2
            linebuffer[m][r][x] = linebuffer[m][r + 1][x];
          }
          // set the last (F-2 index) to the current bottom_row (y + F - 1)
          linebuffer[m][F - 2][x] = input[m][bottom_row][x];
        }
      }

    } // y
  }   // n
}


//----------------------------------------------------------
// Max pooling
//----------------------------------------------------------
// @param[in] : input - input fmaps
//              M - number of input fmaps
//              I - width of input fmaps
// @param[out] : output - output fmaps
template <int M, int I>
void max_pool(bit input[M][I][I], bit output[M][I / 2][I / 2]) {

  for (int m = 0; m < M; m++) {
      for (int y = 0; y < I / 2; y++) {
      #pragma HLS pipeline
    for (int x = 0; x < I / 2; x++) {
        bit max = 0;
        for (int r = 0; r < 2; r++) {
          for (int c = 0; c < 2; c++) {
            if (input[m][2 * y + r][2 * x + c])
              max = 1;
          }
        }
        output[m][y][x] = max;
      }
    }
  }
}

//----------------------------------------------------------
// Flatten the Output from Conv Layer
//----------------------------------------------------------
// @param[in] : input - output fmaps from the last conv layer
// @param[out] : output - input famps of the first dense layer

void flatten(bit input[O_CHANNEL2][O_WIDTH][O_WIDTH], bit output[I_UNITS1]) {
  for (int c = 0; c < O_CHANNEL2; c++) {
    for (int y = 0; y < O_WIDTH; y++) {
      #pragma HLS pipeline
      for (int x = 0; x < O_WIDTH; x++) {
        int o_index = c + (x + y * O_WIDTH) * O_CHANNEL2;
        output[o_index] = input[c][y][x];
      }
    }
  }
}

//----------------------------------------------------------
// Perform Sign Layer
//----------------------------------------------------------
// @param[in] : input - input fmaps
//              M - number of input and output channels
// @param[out] : output - output fmaps

template <int M> void sign(bit16_t input[M], bit output[M]) {
  for (int m = 0; m < M; m++) {
    output[m] = (input[m] > 0) ? 1 : 0;
  }
}

//----------------------------------------------------------
// Perform Argmax Layer
//----------------------------------------------------------
// @param[in] : input - input channels
// @param[out] : output - argmax of the inputs

bit4_t argmax(bit16_t input[NUM_DIGITS]) {
  bit16_t max = input[0];
  bit4_t max_id = 0;
  for (int i = 1; i < NUM_DIGITS; i++) {
    if (input[i] > max) {
      max = input[i];
      max_id = i;
    }
  }
  return max_id;
}

//----------------------------------------------------------
// Perform Dense Layer
//----------------------------------------------------------
// @param[in] : input - input fmaps
//              M - number of input fmaps
//              N - number of output fmaps
//              weight - layer weights
// @param[out] : output - output fmaps

template <int M, int N>
void dense(bit input[M], bit16_t output[N], const bit weight[M][N]) {
  for (int n = 0; n < N; n++) {
    #pragma HLS pipeline
    bit16_t accum = 0;
    for (int m = 0; m < M; m++) {
      accum += input[m] == weight[m][n]; // XNOR
    }
    output[n] = (accum << 1) - M;
  }
}

#endif
