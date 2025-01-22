#pragma once

#include <cuda_runtime.h>

__global__ void residual_forward_naive_kernel(float* out, const float* inp1, const float* inp2, int N) 
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;

    if (t < N) 
    {
      out[t] = inp1[t] + inp2[t];
    }
}

void residual_forward_naive(float* out, const float* inp1, const float* inp2, int N) 
{
  const int block_size = 128;
  const int grid_size = (N - 1) / block_size + 1;
  residual_forward_naive_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
}
