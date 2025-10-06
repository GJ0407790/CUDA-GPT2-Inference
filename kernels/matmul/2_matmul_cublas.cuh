#pragma once

#include "../../utils/cuda_utils.cuh"

constexpr int BLOCK_SIZE = 512;

__global__ void add_bias_kernel(
  float* arr, 
  const float* bias, 
  int B, 
  int T, 
  int OC) 
{
  int bt = blockIdx.y;
  int o = blockIdx.x * blockDim.x + threadIdx.x;

  if (o < OC) 
  {
    arr[bt * OC + o] += bias[o];
  }
}

void add_bias(
  float* arr,
  const float* bias, 
  int B, 
  int T, 
  int OC) 
{
  dim3 gridDim((OC-1)/BLOCK_SIZE+1, B*T);
  add_bias_kernel<<<gridDim, BLOCK_SIZE>>>(arr, bias, B, T, OC);
}

void matmul_forward_cublas(
  float* out, 
  const float* inp, 
  const float* weight, 
  const float* bias,
  int B, 
  int T, 
  int C, 
  int OC) 
{
  // cublas treats arrays as column major
  // A is inp, B is weight^T
  // we use A @ B = (B^T @ A^T)^T
  // reading row as column or column as row is the same as transposing
  // in conclusion we need to feed cublas weight^T and inp
  // or something
  cublasOperation_t transa = CUBLAS_OP_T;
  cublasOperation_t transb = CUBLAS_OP_N;
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm_v2(cublas_handle, transa, transb, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC);
  
  // possible optimization: there might be a matrix format that cublasLt accepts which is faster to translate to from bias
  if (bias != nullptr) 
  {
    add_bias(out, bias, B, T, OC);
  }
}
