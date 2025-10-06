#pragma once

#include <mma.h>
#include <cuda_fp16.h>

constexpr int WARP_SIZE = 32;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;
constexpr int BLOCK_WIDTH = WMMA_K;
constexpr int BLOCK_HEIGHT = WARP_SIZE / WMMA_K;

__global__ void matmul_forward_tensor_kernel(float* __restrict__ out, const float* __restrict__ inp, const float* __restrict__ weight, 
                                             const float* __restrict__ bias, int B, int T, int C, int OC) 
{
  using namespace nvcuda;

  int b = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row_start = blockIdx.y * WMMA_M;
  int col_start = blockIdx.x * WMMA_N;

  inp += (b*T + row_start) * C;
  weight += col_start * C;
  out += (b*T + row_start) * OC + col_start;

  // Similar to tiled matrix multiplication 
  // where thread block collaboratively loads in both inp and weight
  // using shared memory can automatically pad the fragment into desired size
  // 256-bit aligned is the requirement for load_matrix_sync
  __shared__ __align__(256) float out_ins[WMMA_M][WMMA_N];

  // fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> inp_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> weight_frag; // col_major because transposed
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> out_frag;
  // initialize the output fragment
  wmma::fill_fragment(out_frag, 0.0f);
  
  for (int k = 0; k < C; k += WMMA_K)
  {
    wmma::load_matrix_sync(inp_frag, inp, C);
    wmma::load_matrix_sync(weight_frag, weight, C);
    wmma::mma_sync(out_frag, inp_frag, weight_frag, out_frag);

    inp += k;
    weight += k;
  }

  // load back to shared memory
  wmma::store_matrix_sync(&out_ins[0][0], out_frag, WMMA_N, wmma::mem_row_major);

  // Add bias first before storing back to global out
  // Similar to how we load in shared memory, 
  // all threads in a warp collaboratively load 16x16 output elements back to out
  for (int row_offset = 0; row_offset < WMMA_M; row_offset += BLOCK_HEIGHT)
  {
    for (int col_offset = 0; col_offset < WMMA_N; col_offset += BLOCK_WIDTH)
    {
      if ((row_start + row_offset + ty) < T && (col_start + col_offset + tx) < OC)
      {
        if (bias != NULL)
        {
          out_ins[row_offset + ty][col_offset + tx] += bias[col_start + col_offset + tx];
        }

        __stcs(&out[(row_offset + ty)*OC + col_offset + tx], out_ins[row_offset + ty][col_offset + tx]);
      }
    }
  }
}

void matmul_forward_tensor(float* __restrict__ out, const float* __restrict__ inp, const float* __restrict__ weight, 
                           const float* __restrict__ bias, int B, int T, int C, int OC) 
{
  dim3 block_dim(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
  dim3 grid_dim((OC - 1) / WMMA_N + 1, (T - 1) / WMMA_M + 1, B);
  matmul_forward_tensor_kernel<<<grid_dim, block_dim>>>(out, inp, weight, bias, B, T, C, OC);
}
