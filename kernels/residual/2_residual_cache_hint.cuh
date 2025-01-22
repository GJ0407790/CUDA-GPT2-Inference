#pragma once

#include <cuda_runtime.h>

__global__ void residual_forward_cache_hint_kernel(float* out, const float* inp1, const float* inp2, int N) 
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;

    if (t < N) 
    {
      out[t] = __ldcs(&inp1[t]) + __ldcs(&inp2[t]);
    }
}

void residual_forward_cache_hint(float* out, const float* inp1, const float* inp2, int N) 
{
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;
  residual_forward_cache_hint_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
}
