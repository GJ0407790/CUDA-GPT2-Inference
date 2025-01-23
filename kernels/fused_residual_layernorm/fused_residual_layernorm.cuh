#pragma once

__global__ void fused_residual_layernorm_kernel(float* __restrict__ res_out, float* __restrict__ layernorm_out, float* __restrict__ mean, float* __restrict__ rstd,
                                                const float* __restrict__ inp1, const float* __restrict__ inp2, const float* __restrict__ weight,
                                                const float* __restrict__ bias, const int C) 
{
  

  const int tx = threadIdx.x;
  const int row_idx = blockIdx.x;

  auto x1 = reinterpret_cast<const float4*>(inp1 + row_idx * C);
  auto x2 = reinterpret_cast<const float4*>(inp2 + row_idx * C);
  auto ro = reinterpret_cast<float4*>(res_out + row_idx * C);
  auto lo = reinterpret_cast<float4*>(layernorm_out + row_idx * C);

  constexpr int MAX_C = 768;
  constexpr int WARP_SIZE = 32;
  constexpr int STRIDE = 4 * WARP_SIZE;
  constexpr int NUM_ELE_PER_THREAD = MAX_C / (WARP_SIZE * 4);
  float4 eles[NUM_ELE_PER_THREAD]; // each thread need to handle MAX_C/WARP_SIZE float, and float4 contains 4 float

  float sum = 0.0f;
  float sum_of_square = 0.0f;

  for (int ite = 0; ite < NUM_ELE_PER_THREAD; ite++)
  {
    float4 ele1 = x1[tx];
    float4 ele2 = x2[tx];
    
    eles[ite].x = ele1.x + ele2.x;
    eles[ite].y = ele1.y + ele2.y;
    eles[ite].z = ele1.z + ele2.z;
    eles[ite].w = ele1.w + ele2.w;

    __stcs(&ro[tx], eles[ite]);

    sum += eles[ite].x;
    sum += eles[ite].y;
    sum += eles[ite].z;
    sum += eles[ite].w;

    sum_of_square += eles[ite].x * eles[ite].x;
    sum_of_square += eles[ite].y * eles[ite].y;
    sum_of_square += eles[ite].z * eles[ite].z;
    sum_of_square += eles[ite].w * eles[ite].w;

    ro += STRIDE;
    x1 += STRIDE;
    x2 += STRIDE;
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

    __stcs(&lo[tx], eles[ite]);
    lo += STRIDE;
    w += STRIDE;
    b += STRIDE;
  }
}

void fused_residual_layernorm_forward(float* __restrict__ res_out, float* __restrict__ layernorm_out, float* __restrict__ mean, float* __restrict__ rstd,
                                      const float* __restrict__ inp1, const float* __restrict__ inp2, const float* __restrict__ weight,
                                      const float* __restrict__ bias, const int B, const int T, const int C) 
{
  assert(C % 32 == 0);
  // Let a single warp handle a row
  // a block might contain several warps
  // in other words, a block handles multiple rows
  const int block_size = 32;
  fused_residual_layernorm_kernel<<<B*T, block_size>>>(res_out, layernorm_out, mean, rstd, inp1, inp2, weight, bias, C);
}
