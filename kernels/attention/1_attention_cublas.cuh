#pragma once

#include <nvtx3/nvToolsExt.h>

#include "permute.cuh"
#include "../softmax.cuh"
#include "../../utils/cuda_utils.cuh"

void bmm_cublas(
	float* __restrict__ q, 
	float* __restrict__ k, 
	float* __restrict__ inp, 
	int B, int T, int HS, int NH) 
{
	float alpha = 1.0f;
	float beta = 0.0f;
	cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T*HS, q, HS, T*HS, &beta, inp, T, T*T, B*NH);
}

void bmm2_cublas(
	float* __restrict__ att, 
	float* __restrict__ v, 
	float* __restrict__ vac, 
	int B, int T, int HS, int NH) 
{
  float alpha = 1.0f;
	float beta = 0.0f;
	cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T*HS, att, T, T*T, &beta, vac, HS, T*HS, B*NH);
}

void attention_forward_cublas(
	float* __restrict__ out, 
	float* __restrict__ qkvr, 
	float* __restrict__ att,
	const float* __restrict__ inp,
	int B, int T, int C, int NH) 
{
	int HS = C / NH;
 
	float *q = qkvr;
	float *k = qkvr + (B * NH * T * HS);
	float *v = qkvr + 2*(B * NH * T * HS);

	nvtxRangePushA("permute_qkv");
	// permute and separate inp from (B, T, 3, NH, HS) to 3 arrays of (B, NH, T, HS)
	permute(q, k, v, inp, B, T, NH, HS);
	nvtxRangePop();

	nvtxRangePushA("attention");
	// compute pre-attention scores (q @ k.transpose)
	float* preatt = const_cast<float*>(inp); // reuse input buffer for preatt
	bmm_cublas(q, k, preatt, B, T, HS, NH);

	// compute the softmax of the preattention scores
	// preatt[B][NH][T][T] -> att[B][NH][T][T]
  float scale = 1.0 / sqrtf(HS);
	softmax_forward(att, scale, preatt, B * NH, T);

	// compute vaccum values (att @ v) = (B, NH, T, T) @ (B, NH, T, HS) -> (B, NH, T, HS)
	float* vac = const_cast<float*>(inp);
	bmm2_cublas(att, v, vac, B, T, HS, NH);
	nvtxRangePop();

	nvtxRangePushA("unpermute_out");
	// unpermute from vac[B][NH][N][d] to out[B][N][NH][d]
	unpermute(vac, out, B, T, NH, HS);
	nvtxRangePop();
}