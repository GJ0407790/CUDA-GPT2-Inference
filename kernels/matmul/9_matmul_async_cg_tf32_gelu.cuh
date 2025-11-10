#pragma once

#include "ptx.cuh"

float __device__ __forceinline__ gelu(float x)
{
	const float SCALING_FACTOR = sqrtf(2.0f / M_PI);

	const float cube = 0.044715f * x * x * x;
	return 0.5f * x * (1.0f + tanhf(SCALING_FACTOR * (x + cube)));
}

template<const int BM,
         const int BN,
         const int BK,
         const int WM,
         const int WN,
         const int THREADS_PER_BLOCK>
__launch_bounds__(THREADS_PER_BLOCK) __global__ void matmul_forward_async_cg_tf32_gelu_kernel(
	float* __restrict__ out,
	const float* __restrict__ inp,
	const float* __restrict__ weight,
	const float* __restrict__ bias,
	const int T,
	const int C,
	const int OC,
	const bool is_gelu) 
{
	const int tid = threadIdx.x;
	const int quad_id = tid & 3;
	const int warp_id = tid / 32 ;

	const int warp_col = warp_id % (BN / WN);
	const int warp_row = warp_id / (BN / WN);

	__shared__ float input_tile[2][BM * BK];
	__shared__ float weight_tile[2][BN * BK];

	inp += blockIdx.y * BM * C;
	weight += blockIdx.x * BN * C;
	out += (blockIdx.y * BM + warp_row * WM) * OC + blockIdx.x * BN + warp_col * WN;

	// for loading in input and weight tiles (gmem to smem)
	const int stride = blockDim.x / (BK / 4);
	const int inner_row = tid / (BK / 4);
	const int inner_col = (tid % (BK / 4)) * 4; // each thread can load in 4 elements, hence BK/4 threads needed per row

	// The fragment is m16-n16-k8
	// Each warp will handle WMxWN output tile
	// There's (WM/16) * (WN/16) fragments, with each fragment needing 8 registers
	// Hence, total output registers needed per thread = (WM/16) * (WN/16) *8
	float c[WM/16][WN/16][8] = {0.0f};
	Reg32 as[WM/16][BK/8][4]; // read input at once and reuse
	Reg32 bs[4];

	// load in bias into c as initial values
	if (bias != nullptr)
	{
		bias += blockIdx.x * BN + warp_col * WN;
		auto bias_f2 = reinterpret_cast<const float2*>(bias);

		// every 16 columns need to load in 4 biases
		// there are WM/16 fragments that share the same biases
		#pragma unroll
		for (int n = 0; n < WN/16; n++, bias_f2 += 8)
		{
			float2 bias0 = bias_f2[quad_id];
			float2 bias1 = bias_f2[quad_id + 4];

			#pragma unroll
			for (int m = 0; m < WM/16; m++)
			{
				c[m][n][0] = bias0.x;
				c[m][n][1] = bias0.y;
				c[m][n][2] = bias0.x;
				c[m][n][3] = bias0.y;

				c[m][n][4] = bias1.x;
				c[m][n][5] = bias1.y;
				c[m][n][6] = bias1.x;
				c[m][n][7] = bias1.y;
			}
		}
	}

	#pragma unroll
	for (int offset = 0; offset < BM; offset += stride)
	{
		cp_async_16B(
			reinterpret_cast<void*>(input_tile[0] + (offset + inner_row) * BK + inner_col),
			reinterpret_cast<const void*>(inp + (offset + inner_row) * C + inner_col)
		);
	}

	#pragma unroll
	for (int offset = 0; offset < BN; offset += stride)
	{
		cp_async_16B(
			reinterpret_cast<void*>(weight_tile[0] + (offset + inner_row) * BK + inner_col),
			reinterpret_cast<const void*>(weight + (offset + inner_row) * C + inner_col)
		);
	}

	cp_async_commit_group();

	for (int k = 0; k < C / BK; k++)
	{
		// load in input and weights
		if (k + 1 < C / BK)
		{
			inp += BK;
			weight += BK;

			#pragma unroll
			for (int offset = 0; offset < BM; offset += stride)
			{
				cp_async_16B(
					reinterpret_cast<void*>(input_tile[(k+1) % 2] + (offset + inner_row) * BK + inner_col),
					reinterpret_cast<const void*>(inp + (offset + inner_row) * C + inner_col)
				);
			}

			#pragma unroll
			for (int offset = 0; offset < BN; offset += stride)
			{
				cp_async_16B(
					reinterpret_cast<void*>(weight_tile[(k+1) % 2] + (offset + inner_row) * BK + inner_col),
					reinterpret_cast<const void*>(weight + (offset + inner_row) * C + inner_col)
				);
			}

			cp_async_commit_group();
			cp_async_wait_group1();
		}
		else
		{
			cp_async_wait_group0(); // there is only 1 cg left
		}
		
		__syncthreads();

		// mma
		// load in inp once
		#pragma unroll
		for (int m = 0; m < WM/16; m++)
		{
			#pragma unroll
			for (int b = 0; b < BK/8; b++)
			{
				wmma_tf32_ldmatrix_m16n16k8<true /*is_A*/>(
					&input_tile[(k % 2)][(warp_row * WM + m * 16) * BK + b * 8], // all points to the start of the tile
					as[m][b][0], as[m][b][1], as[m][b][2], as[m][b][3],
					BK // stride
				);
			}
		}

		// auto as_f32 = reinterpret_cast<float*>(as);
		// printf("[%d]: {%f, %f, %f, %f, %f, %f, %f, %f}\n", tid, 
		// 	as_f32[0], as_f32[1], as_f32[2], as_f32[3], 
		// 	as_f32[4], as_f32[5], as_f32[6], as_f32[7]);

		// need to iterate BN using the same as registers
		#pragma unroll
		for (int n = 0; n < WN/16; n++)
		{
			#pragma unroll
			for (int b = 0; b < BK/8; b++)
			{
				wmma_tf32_ldmatrix_m16n16k8<false /*is_A*/>(
					&weight_tile[(k % 2)][(warp_col * WN + n * 16) * BK + b * 8], // all points to the start of the tile
					bs[0], bs[1], bs[2], bs[3],
					BK // stride
				);

				// auto bs_f32 = reinterpret_cast<float*>(bs);
				// printf("[%d]: {%f, %f, %f, %f}\n", tid, 
				// 	bs_f32[0], bs_f32[1], bs_f32[2], bs_f32[3]);

				#pragma unroll												 
				for (int m = 0; m < WM/16; m++)
				{
					wmma_tf32_mma_m16n16k8(
						as[m][b][0], as[m][b][1], as[m][b][2], as[m][b][3],
						bs[0], bs[1], bs[2], bs[3],
						c[m][n][0], c[m][n][1], c[m][n][2], c[m][n][3],
						c[m][n][4], c[m][n][5], c[m][n][6], c[m][n][7]
					);
				}
			}
		}

		__syncthreads();
	}

	// apply activation if needed
	if (is_gelu)
	{
		#pragma unroll
		for (int m = 0; m < WM/16; m++)
		{
			#pragma unroll
			for (int n = 0; n < WN/16; n++)
			{
				#pragma unroll
				for (int r = 0; r < 8; r++)
				{
					c[m][n][r] = gelu(c[m][n][r]);
				}
			}
		}
	}

	// store back to output
	Reg32* c_u32 = reinterpret_cast<Reg32*>(c);

	#pragma unroll
	for (int n = 0; n < WN/16; n++)
	{
		#pragma unroll
		for (int m = 0; m < WM/16; m++)
		{
			wmma_f32_stmatrix_gmem_m16n16k8(
				&out[m*16*OC],
				c_u32[(m * (WN/16) + n) * 8 + 0], c_u32[(m * (WN/16) + n) * 8 + 1],
				c_u32[(m * (WN/16) + n) * 8 + 2], c_u32[(m * (WN/16) + n) * 8 + 3],
				c_u32[(m * (WN/16) + n) * 8 + 4], c_u32[(m * (WN/16) + n) * 8 + 5],
				c_u32[(m * (WN/16) + n) * 8 + 6], c_u32[(m * (WN/16) + n) * 8 + 7],
				OC // stride
			);
		}

		out += 16; // move to next set of columns
	}
}

void matmul_forward_async_cg_tf32_gelu(
	float* __restrict__ out, 
	const float* __restrict__ inp, 
	const float* __restrict__ weight, 
  const float* __restrict__ bias, 
	const int B, 
	const int T, 
	const int C, 
	const int OC,
	bool is_gelu) 
{
  // Each block computes a BM x BN output tile
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16;
  constexpr int WM = 64;
  constexpr int WN = 64;

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

  matmul_forward_async_cg_tf32_gelu_kernel<BM, BN, BK, WM, WN, THREADS_PER_BLOCK>
						   						    			 		  <<<grid_dim, block_dim>>>(out, inp, weight, bias, T, C, OC, is_gelu);
}
