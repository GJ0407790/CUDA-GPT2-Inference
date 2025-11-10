#pragma once

#include "ptx.cuh"

template<const int BM,
         const int BN,
         const int BK,
         const int WM,
         const int WN,
         const int THREADS_PER_BLOCK>
__launch_bounds__(THREADS_PER_BLOCK) __global__ void matmul_forward_ptx_2d_kernel(
	float* __restrict__ out,
	const float* __restrict__ inp,
	const float* __restrict__ weight,
	const float* __restrict__ bias,
	const int T,
	const int C,
	const int OC) 
{
	const int tid = threadIdx.x;
	const int quad_id = tid & 3;
	const int warp_id = tid / 32 ;
	const int lane_id = tid % 32;

	const int warp_col = warp_id % (BN / WN);
	const int warp_row = warp_id / (BN / WN);

	__shared__ bf16 input_tile[BM * BK];
	__shared__ bf16 weight_tile[BN * BK];

	inp += blockIdx.y * BM * C;
	weight += blockIdx.x * BN * C;
	out += (blockIdx.y * BM + warp_row * WM) * OC + blockIdx.x * BN + warp_col * WN;

	// for loading in input and weight tiles (gmem to smem)
	const int stride = blockDim.x / (BK / 4);
	const int inner_row = tid / (BK / 4);
	const int inner_col = (tid % (BK / 4)) * 4; // each thread can load in 4 elements, hence BK/4 threads needed per row

	// for loading in fragments (smem to reg)
	const int frag_row = lane_id % 16;
	const int frag_col = (lane_id / 16) * 8;

	// The fragment is m16-n8-k16
	// Each warp will handle WMxWN output tile
	// There's (WM/16) * (WN/8) fragments, with each fragment needing 4 registers
	// Hence, total output registers needed per thread = (WM/16) * (WN/8) *4
	float c[WM/16][WN/8][4] = {0.0f};
	Reg32 as[WM/16][4]; // read input at once and reuse
	Reg32 bs[4];

	// load in bias into c as initial values
	if (bias != nullptr)
	{
		bias += blockIdx.x * BN + warp_col * WN;
		auto bias_f2 = reinterpret_cast<const float2*>(bias);

		// every 8 columns need to load in 2 biases
		// there are WM/16 fragments that share the same biases
		#pragma unroll
		for (int n = 0; 8*n < WN; n++, bias_f2 += 4)
		{
			float2 bs = bias_f2[quad_id];

			#pragma unroll
			for (int m = 0; m < WM/16; m++)
			{
				// c[m][n][0] and c[m][n][2] are the same column
				// same for c[m][n][1] and c[m][n][3]
				c[m][n][0] = bs.x;
				c[m][n][1] = bs.y;
				c[m][n][2] = bs.x;
				c[m][n][3] = bs.y;
			}
		}
	}

	for (int k = 0; k < C / BK; k++)
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
		#pragma unroll
		for (int m = 0; m < WM/16; m++)
		{
			ldmatrix_m8n8x4<false>(&input_tile[(warp_row * WM + m * 16 + frag_row) * BK + frag_col], 
										 		 	   as[m][0], as[m][1], as[m][2], as[m][3]);
		}
		
		// need to iterate BN using the same as registers
		#pragma unroll
		for (int n = 0; n < WN/16; n++)
		{
			ldmatrix_m8n8x4<false>(&weight_tile[(warp_col * WN + n * 16 + frag_row) * BK + frag_col], 
										 		 		 bs[0], bs[1], bs[2], bs[3]);

			#pragma unroll												 
			for (int m = 0; m < WM/16; m++)
			{
				// for 16 columns, 
				// c[m][n] stores the result for first 8 columns whereas
				// c[m][n+1] stores the result for the next 8 columns
				mma_m16n8k16(as[m][0], as[m][1], as[m][2], as[m][3],
										 bs[0], bs[2], // flipped cause col-major
										 c[m][2*n][0], c[m][2*n][1], c[m][2*n][2], c[m][2*n][3]);

				mma_m16n8k16(as[m][0], as[m][1], as[m][2], as[m][3],
										 bs[1], bs[3],
										 c[m][2*n+1][0], c[m][2*n+1][1], c[m][2*n+1][2], c[m][2*n+1][3]);
			}
		}

		__syncthreads();
	}

	// store back to output
	out += (lane_id/4) * OC + (lane_id%4) * 2;

	#pragma unroll
	for (int n = 0; n < WN/8; n++)
	{
		#pragma unroll
		for (int m = 0; m < WM/16; m++)
		{
			out[(m * 16) * OC] = c[m][n][0];
			out[(m * 16) * OC + 1] = c[m][n][1];

			out[(m * 16 + 8) * OC] = c[m][n][2];
			out[(m * 16 + 8) * OC + 1] = c[m][n][3];
		}

		out += 8;
	}
}

void matmul_forward_ptx_2d(
	float* __restrict__ out, 
	const float* __restrict__ inp, 
	const float* __restrict__ weight, 
  const float* __restrict__ bias, 
	const int B, 
	const int T, 
	const int C, 
	const int OC) 
{
  // Each block computes a BM x BN output tile
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16;
  constexpr int WM = 64;
  constexpr int WN = 32;

  assert(BM % WM == 0);
  assert(BN % WN == 0);

  // For Supermma, maximum m-n-k is 16-8-16 for bf16
  // need a warp for every 16 rows of BM
  assert(WM % 16 == 0);
  assert(WN % 16 == 0);

  constexpr int NUM_WARPS = (BM / WM) * (BN / WN);
  constexpr int THREADS_PER_BLOCK = 32 * NUM_WARPS;

	assert(C % BK == 0);
	assert(OC % BN == 0);
	assert((B * T) % BM == 0);

  dim3 grid_dim((OC - 1) / BN + 1, (B * T - 1) / BM + 1);
	int block_dim = THREADS_PER_BLOCK;

  matmul_forward_ptx_2d_kernel<BM, BN, BK, WM, WN, THREADS_PER_BLOCK>
						   						    <<<grid_dim, block_dim>>>(out, inp, weight, bias, T, C, OC);
}
