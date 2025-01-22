#pragma once

#include <math.h>
#include <assert.h>
#include <float.h>

__global__ void layernorm_forward_block_vectorized_kernel(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                               const float* __restrict__ inp, const float* __restrict__ weight,
                                               const float* __restrict__ bias, const int C) 
{
  // A block handles a single row
  const int row_idx = blockIdx.x;
  const int tx = threadIdx.x;
  auto x = reinterpret_cast<const float4*>(inp + row_idx * C);
  auto o = reinterpret_cast<float4*>(out + row_idx * C);

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
  
  float4 ele = x[tx];

  local_sum += ele.x;
  local_sum += ele.y;
  local_sum += ele.z;
  local_sum += ele.w;

  local_sum_of_square += ele.x * ele.x;
  local_sum_of_square += ele.y * ele.y;
  local_sum_of_square += ele.z * ele.z;
  local_sum_of_square += ele.w * ele.w;

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

  auto w = reinterpret_cast<const float4*>(weight);
  auto b = reinterpret_cast<const float4*>(bias);

  float4 weights = w[tx];
  float4 biases = b[tx];

  ele.x = s * weights.x * (ele.x - m) + biases.x;
  ele.y = s * weights.y * (ele.y - m) + biases.y;
  ele.z = s * weights.z * (ele.z - m) + biases.z;
  ele.w = s * weights.w * (ele.w - m) + biases.w;

  __stcs(&o[tx], ele);
}

void layernorm_forward_block_vectorized(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                        const float* __restrict__ inp, const float* __restrict__ weight,
                                        const float* __restrict__ bias, const int B, const int T, const int C) 
{
  assert(C % 4 == 0);
  // Let each thread handle 4 elements
  const int block_size = C / 4;
  layernorm_forward_block_vectorized_kernel<<<B*T, block_size>>>(out, mean, rstd, inp, weight, bias, C);
}