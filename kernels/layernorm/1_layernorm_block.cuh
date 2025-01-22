#pragma once

#include <math.h>
#include <assert.h>
#include <float.h>

__global__ void layernorm_forward_block_kernel(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                               const float* __restrict__ inp, const float* __restrict__ weight,
                                               const float* __restrict__ bias, const int C) 
{
  // A block handles a single row
  const int row_idx = blockIdx.x;
  const float* x = inp + row_idx * C;
  const int tx = threadIdx.x;

  __shared__ float sum; // Sigma(X)
  __shared__ float sum_of_square; // Sigma(X^2)

  if (tx == 0)
  {
    // let the first thread initialize the shared value
    sum = 0.0f;
    sum_of_square = 0.0f;
  }

  // reduce to local variable first before atomic add to the shared variable
  float local_sum = 0.0f;
  float local_sum_of_square = 0.0f;

  __syncthreads(); // make sure shared variables are initialized

  for (int i = 0; i < 4; i++)
  {
    float ele = x[tx + i * blockDim.x];
    local_sum += ele;
    local_sum_of_square += ele * ele;
  }

  // reduce to shared variable
  atomicAdd(&sum, local_sum);
  atomicAdd(&sum_of_square, local_sum_of_square);

  __syncthreads();

  float m = sum / C; // mean
  float s = 1.0f / sqrt((sum_of_square / C) - m*m + 1e-5f); // std; 1e-5 is for numerical stability

  if (tx == 0)
  {
    __stcs(&mean[row_idx], m);
    __stcs(&rstd[row_idx], s);
  }

  for (int i = 0; i < 4; i++)
  {
    // rely on L1/L2 cache
    float n = s * (x[tx + i * blockDim.x] - m);
    float output_ele = n * weight[tx + i * blockDim.x] + bias[tx + i * blockDim.x];
    __stcs(&out[row_idx * C + tx + i * blockDim.x], output_ele);
  }
}

void layernorm_forward_block(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                             const float* __restrict__ inp, const float* __restrict__ weight,
                             const float* __restrict__ bias, const int B, const int T, const int C) 
{
  assert(C % 4 == 0);
  // Let each thread handle 4 elements
  const int block_size = C / 4;
  layernorm_forward_block_kernel<<<B*T, block_size>>>(out, mean, rstd, inp, weight, bias, C);
}