#pragma once

#include <math.h>
#include <assert.h>
#include <float.h>

__global__ void layernorm_forward_warp_kernel(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                                               const float* __restrict__ inp, const float* __restrict__ weight,
                                               const float* __restrict__ bias, const int C) 
{
  const int row_idx = blockIdx.x;
  const int tx = threadIdx.x;

  auto x = reinterpret_cast<const float4*>(inp + row_idx * C);
  auto o = reinterpret_cast<float4*>(out + row_idx * C);

  constexpr int MAX_C = 768;
  constexpr int WARP_SIZE = 32;
  constexpr int STRIDE = 4 * WARP_SIZE;
  constexpr int NUM_ELE_PER_THREAD = MAX_C / (WARP_SIZE * 4);
  float4 eles[NUM_ELE_PER_THREAD]; // each thread need to handle MAX_C/WARP_SIZE float, and float4 contains 4 float

  float sum = 0.0f;
  float sum_of_square = 0.0f;

  for (int ite = 0; ite < NUM_ELE_PER_THREAD; ite++)
  {
    eles[ite] = x[tx];

    sum += eles[ite].x;
    sum += eles[ite].y;
    sum += eles[ite].z;
    sum += eles[ite].w;

    sum_of_square += eles[ite].x * eles[ite].x;
    sum_of_square += eles[ite].y * eles[ite].y;
    sum_of_square += eles[ite].z * eles[ite].z;
    sum_of_square += eles[ite].w * eles[ite].w;

    x += STRIDE;
  }

  // warp reduce
  for (int stride = 16; stride >= 1; stride >>= 1) {
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, stride);
    sum_of_square += __shfl_xor_sync(0xFFFFFFFF, sum_of_square, stride);
  }

  float m = sum / C; // mean
  float s = 1.0f / sqrt((sum_of_square / C) - m*m + 1e-5f); // std; 1e-5 is for numerical stability

  if (tx == 0)
  {
    __stcs(&mean[row_idx], m);
    __stcs(&rstd[row_idx], s);
  }

  auto w = reinterpret_cast<const float4*>(weight);
  auto b = reinterpret_cast<const float4*>(bias);

  for (int ite = 0; ite < NUM_ELE_PER_THREAD; ite++)
  {
    float4 weights = w[tx];
    float4 biases = b[tx];

    eles[ite].x = s * weights.x * (eles[ite].x - m) + biases.x;
    eles[ite].y = s * weights.y * (eles[ite].y - m) + biases.y;
    eles[ite].z = s * weights.z * (eles[ite].z - m) + biases.z;
    eles[ite].w = s * weights.w * (eles[ite].w - m) + biases.w;

    __stcs(&o[tx], eles[ite]);
    o += STRIDE;
    w += STRIDE;
    b += STRIDE;
  }
}

void layernorm_forward_warp(float* __restrict__ out, float* __restrict__ mean, float* __restrict__ rstd,
                            const float* __restrict__ inp, const float* __restrict__ weight,
                            const float* __restrict__ bias, const int B, const int T, const int C) 
{
  assert(C % 32 == 0);
  // Launch a single warp for each row
  const int block_size = 32;
  layernorm_forward_warp_kernel<<<B*T, block_size>>>(out, mean, rstd, inp, weight, bias, C);
}