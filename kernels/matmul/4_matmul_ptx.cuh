#pragma once

#include "ptx.cuh"

// Each block computes a BM x BN output tile
constexpr int BM = 128;
constexpr int BN = 64;
constexpr int BK = 16;

// For Supermma, maximum m-n-k is 16-8-16 for bf16
// need a warp for every 16 rows of BM
constexpr int SUPERMMA_DIM = 16;
constexpr int NUM_WARPS = BM / SUPERMMA_DIM;
constexpr int THREADS_PER_BLOCK = 32 * NUM_WARPS;

template<const int BM,
         const int BN,
         const int BK>
__launch_bounds__(THREADS_PER_BLOCK) __global__ void matmul_forward_ptx_kernel(
	float* __restrict__ out, 
	const float* __restrict__ inp, 
	const float* __restrict__ weight, 
	const float* __restrict__ bias, 
	const int T, 
	const int C, 
	const int OC) 
{
	assert(BK == 16);
	
	const int tid = threadIdx.x;
	const int quad_id = tid & 3;
	const int warp_id = tid / 32 ;
	const int lane_id = tid % 32;

	__shared__ bf16 input_tile[BM * BK];
	__shared__ bf16 weight_tile[BN * BK];

	inp += blockIdx.y * BM * C;
	weight += blockIdx.x * BN * C;
	out += blockIdx.y * BM * OC + blockIdx.x * BN;

	// for loading in input and weight tiles (gmem to smem)
	const int stride = blockDim.x / (BK / 4);
	const int inner_row = threadIdx.x / (BK / 4);
	const int inner_col = quad_id * 4; // each thread can load in 4 elements

	// for loading in fragments (smem to reg)
	const int frag_row = lane_id % 16;
	const int frag_col = (lane_id / 16) * 8;

	// The fragment is m16-n8-k16
	// Each warp will handle 16xBN output tile
	// There's BN/8 fragments, with each fragment needing 4 registers
	// Hence, total output registers needed per thread = (BN/8)*4
	float c[BN / 2] = {0.0f};
	Reg32 as[4]; // 8 bf16 values
	Reg32 bs[4];

	// load in bias into c as initial values
	if (bias != nullptr)
	{
		bias += blockIdx.x * BN;
		auto bias_f2 = reinterpret_cast<const float2*>(bias);
		
		// every 8 columns need to load in 2 biases
		#pragma unroll
		for (int n = 0; 8*n < BN; n++, bias_f2 += 4)
		{
			float2 bs = bias_f2[quad_id];

			c[4*n] = bs.x;
			c[4*n + 1] = bs.y;
			c[4*n + 2] = bs.x;
			c[4*n + 3] = bs.y;
		}
	}

	for (int k = 0; k < C; k += BK)
	{
		// load in input and weights
		#pragma unroll
		for (int offset = 0; offset < BM; offset += stride)
		{
			const float4* inp_f4 = reinterpret_cast<const float4*>(inp + (offset + inner_row) * C + inner_col);
			float4 in_4 = inp_f4[0];

			input_tile[(offset + inner_row) * BK + inner_col + 0] = __nv_bfloat16(in_4.x);
			input_tile[(offset + inner_row) * BK + inner_col + 1] = __nv_bfloat16(in_4.y);
			input_tile[(offset + inner_row) * BK + inner_col + 2] = __nv_bfloat16(in_4.z);
			input_tile[(offset + inner_row) * BK + inner_col + 3] = __nv_bfloat16(in_4.w);
		}

		#pragma unroll
		for (int offset = 0; offset < BN; offset += stride)
		{
			const float4* weight_f4 = reinterpret_cast<const float4*>(weight + (offset + inner_row) * C + inner_col);
			float4 w_4 = weight_f4[0];

			weight_tile[(offset + inner_row) * BK + inner_col + 0] = __nv_bfloat16(w_4.x);
			weight_tile[(offset + inner_row) * BK + inner_col + 1] = __nv_bfloat16(w_4.y);
			weight_tile[(offset + inner_row) * BK + inner_col + 2] = __nv_bfloat16(w_4.z);
			weight_tile[(offset + inner_row) * BK + inner_col + 3] = __nv_bfloat16(w_4.w);
		}

		__syncthreads();

		inp += BK;
		weight += BK;

		// mma
		// load in inp once
		ldmatrix_m16n16x4<false>(&input_tile[(warp_id*16 + frag_row) * BK + frag_col], as[0], as[1], as[2], as[3]);

		// need to iterate BN using the same as registers
		#pragma unroll
		for (int n = 0; n < BN; n += 16)
		{
			ldmatrix_m16n16x4<false>(&weight_tile[(n + frag_row) * BK + frag_col], bs[0], bs[1], bs[2], bs[3]);

			// every 16 n requires 8 register
			// hence (n/16) * 8 = n/2 stride
			float* curr_c = &c[n/2];
			mma_m16n8k16(as[0], as[1], as[2], as[3],
									 bs[0], bs[2], // flipped cause col-major
									 curr_c[0], curr_c[1], curr_c[2], curr_c[3]);

			mma_m16n8k16(as[0], as[1], as[2], as[3],
									 bs[1], bs[3],
									 curr_c[4], curr_c[5], curr_c[6], curr_c[7]);
		}

		__syncthreads();
	}

	// store back to output
	out += (warp_id*16 + lane_id/4) * OC + (lane_id%4) * 2;

	#pragma unroll
	for (int n = 0; n < BN; n += 8)
	{
		out[0] = c[n/2 + 0];
		out[1] = c[n/2 + 1];

		out[8*OC + 0] = c[n/2 + 2];
		out[8*OC + 1] = c[n/2 + 3];

		out += 8;
	}
}

void matmul_forward_ptx(
	float* __restrict__ out, 
	const float* __restrict__ inp, 
	const float* __restrict__ weight, 
  const float* __restrict__ bias, 
	const int B, 
	const int T, 
	const int C, 
	const int OC) 
{
	assert(C % BK == 0);
	assert(OC % BN == 0);
	assert((B * T) % BM == 0);

  dim3 grid_dim((OC - 1) / BN + 1, (B * T - 1) / BM + 1);
	int block_dim = THREADS_PER_BLOCK;

  matmul_forward_ptx_kernel<BM, BN, BK>
						   						 <<<grid_dim, block_dim>>>(out, inp, weight, bias, T, C, OC);
}
