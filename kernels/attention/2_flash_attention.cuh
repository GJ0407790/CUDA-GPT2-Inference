#pragma once

#include <nvtx3/nvToolsExt.h>

#include "permute.cuh"
#include "../matmul/ptx.cuh"

template<const int Bc,
				 const int Br,
				 const int Wr,
				 const int HS,
				 const int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ void flash_attention_kernel(
	float* __restrict__ out,
	const float* __restrict__ q,
	const float* __restrict__ k,
	const float* __restrict__ v,
	const int T, const float inv_temperature
)
{
	assert(Br == Bc);
	assert(Bc <= HS); // we use q_tile to store S temporarily

	const int tid = threadIdx.x;	
	const int i = blockIdx.y;
	const int warp_id = tid / 32;
	const int lane_id = tid % 32;

	const int frag_row = lane_id / 4;
	const int frag_col = (lane_id % 4) * 2;

	__shared__ float k_tile[Bc][HS];
	__shared__ float v_tile[Bc][HS];
	__shared__ float q_tile[Br][HS];

	// each warp handles 16 rows, following the tensor core matmul layout
	// each thread stores 2 rows of output
	// 4 registers per row is store consecutively
	float o_reg[HS/16][2][4] = {0.0f}; // o_reg = softmax(q @ k.T) @ V
	float m[2] = {-FLT_MAX}; 
	float l[2] = {0.0f};

	Reg32 qs[HS/8][4]; // load q into registers
	Reg32 vks[4]; // register for loading k and v

	out += (blockIdx.x * T + i * Br + warp_id * Wr) * HS;
	q += (blockIdx.x * T + i * Br) * HS;
	k += blockIdx.x * T * HS;
	v += blockIdx.x * T * HS;

	const int stride = blockDim.x / (HS / 4); // each thread can read/write 4 floats at a time
	const int inner_row = tid / (HS / 4);
	const int inner_col = (tid % (HS / 4)) * 4;

	// load Q block
	#pragma unroll
	for (int offset = 0; offset < Br; offset += stride)
	{
		auto q4 = reinterpret_cast<const float4*>(&q[(inner_row + offset) * HS + inner_col]);
		reinterpret_cast<float4*>(&q_tile[inner_row + offset][inner_col])[0] = q4[0];
	}

	// wait for Q to be loaded
	__syncthreads();

	// for loading in fragments (smem to reg)
	// load q into registers
	#pragma unroll
	for (int h = 0; h < HS/8; h++)
	{
		wmma_tf32_ldmatrix_m16n16k8<true/*is_A*/>(&q_tile[warp_id * 16][h * 8], 
										 	 					 							qs[h][0], qs[h][1], qs[h][2], qs[h][3],
																 							HS); // stride
	}

	for (int j = 0; j <= i; j++)
	{
		// load in K and V block
		#pragma unroll
		for (int offset = 0; offset < Bc; offset += stride)
		{
			auto k4 = reinterpret_cast<const float4*>(&k[(inner_row + offset) * HS + inner_col]);
			reinterpret_cast<float4*>(&k_tile[inner_row + offset][inner_col])[0] = k4[0];
		}

		#pragma unroll
		for (int offset = 0; offset < Bc; offset += stride)
		{
			auto v4 = reinterpret_cast<const float4*>(&v[(inner_row + offset) * HS + inner_col]);
			reinterpret_cast<float4*>(&v_tile[inner_row + offset][inner_col])[0] = v4[0];
		}

		k += Bc * HS;
		v += Bc * HS;

		__syncthreads();

		// compute attention S = Q @ K.T
		// S is of shape (Br, Bc), every 16 columns need 8 register to store
		float s[Bc/16][2][4] = {0.0f}; // (col_frag, row, register)

		#pragma unroll
		for (int c = 0; c < Bc/16; c++)
		{
			#pragma unroll
			for (int h = 0; h < HS/8; h++)
			{
				wmma_tf32_ldmatrix_m16n16k8<false/*is_A*/>(&k_tile[c * 16][h * 8], 
													 												 vks[0], vks[1], vks[2], vks[3],
																									 HS); // stride
				
				wmma_tf32_mma_m16n16k8(
					qs[h][0], qs[h][1], qs[h][2], qs[h][3],
					vks[0], vks[1], vks[2], vks[3],
					s[c][0][0], s[c][0][1], s[c][1][0], s[c][1][1],
					s[c][0][2], s[c][0][3], s[c][1][2], s[c][1][3]
				);
			}
		}
		
		// compute m, softmax P and l
		float new_m[2] = {m[0], m[1]};

		#pragma unroll
		for (int c = 0; c < Bc/16; c++)
		{
			const int cpos = c * 16 + frag_col;
			
			#pragma unroll
			for (int r = 0; r < 2; r++)
			{
				// each thread handles 2 rows
				const int rpos = warp_id * 16 + frag_row + r * 8;

				#pragma unroll
				for (int t = 0; t < 4; t++) 
				{
					const int curr_pos = cpos + (t/2) * 8 + (t & 1);

					if ((i == j) && (curr_pos > rpos)) // diagonal block
					{
						// causal mask out of bound
						s[c][r][t] = 0.0f; // set to 0.0f after softmax
					}
					else
					{
						// in bound, compute new_m
						new_m[r] = fmaxf(new_m[r], s[c][r][t]);
					}
				}
			}
		}

		// reduce new_m across quad
		constexpr unsigned int MASK = 0xFFFFFFFF;

		// need 2 shuffles to reduce within a quad
		new_m[0] = fmaxf(new_m[0], __shfl_xor_sync(MASK, new_m[0], 2));
		new_m[0] = fmaxf(new_m[0], __shfl_xor_sync(MASK, new_m[0], 1));
		new_m[1] = fmaxf(new_m[1], __shfl_xor_sync(MASK, new_m[1], 2));
		new_m[1] = fmaxf(new_m[1], __shfl_xor_sync(MASK, new_m[1], 1));

		// compute l and softmax P
		// first normalize l and o for second iteration and beyond
		if (j > 0)
		{
			#pragma unroll
			for (int r = 0; r < 2; r++)
			{
				float m_normalize = __expf(inv_temperature * (m[r] - new_m[r]));

				l[r] *= m_normalize;

				#pragma unroll
				for (int h = 0; h < HS/16; h++)
				{
					#pragma unroll
					for (int t = 0; t < 4; t++)
					{
						o_reg[h][r][t] *= m_normalize;
					}
				}
			}
		}
	
		// update the max m
		m[0] = new_m[0];
		m[1] = new_m[1];

		// Compute softmax P and accumulate l
		float new_l[2] = {0.0f, 0.0f};
		#pragma unroll
		for (int c = 0; c < Bc/16; c++)
		{
			const int cpos = c * 16 + frag_col;
			
			#pragma unroll
			for (int r = 0; r < 2; r++)
			{
				const int rpos = warp_id * 16 + frag_row + r * 8;
				float p_sum = 0.0f; // accumulate on a register first to preserve precision

				#pragma unroll
				for (int t = 0; t < 4; t++) 
				{
					const int curr_pos = cpos + (t/2) * 8 + (t & 1);
					if ((i != j) || (curr_pos <= rpos))
					{
						s[c][r][t] = __expf(inv_temperature * (s[c][r][t] - m[r]));
						p_sum += s[c][r][t];
					}
				}

				new_l[r] += p_sum;
			}
		}

		// shuffle the l across the warp
		new_l[0] += __shfl_xor_sync(MASK, new_l[0], 2);
		new_l[0] += __shfl_xor_sync(MASK, new_l[0], 1);
		new_l[1] += __shfl_xor_sync(MASK, new_l[1], 2);
		new_l[1] += __shfl_xor_sync(MASK, new_l[1], 1);

		l[0] += new_l[0];
		l[1] += new_l[1];

		// store S back to smem for PV computation because the register layout is different
		// use q_tile as buffer
		#pragma unroll
		for (int c = 0; c < Bc/16; c++)
		{
			Reg32* s_u32 = reinterpret_cast<Reg32*>(&s[c][0][0]);

			wmma_f32_stmatrix_smem_m16n16k8(
				&q_tile[warp_id * 16][c * 16], 
				s_u32[0], s_u32[1], s_u32[4], s_u32[5],
				s_u32[2], s_u32[3], s_u32[6], s_u32[7],
				HS // stride
			);
		}

		// compute O = O + PV
		// P is (16, BC), V is (BC, HS)
		Reg32 ps[4];

		#pragma unroll
		for (int c = 0; c < Bc/8; c++)
		{
			wmma_tf32_ldmatrix_m16n16k8<true /*is_A*/>
																 (
																	&q_tile[warp_id * 16][c * 8], 
										 	 					 	ps[0], ps[1], ps[2], ps[3],
																 	HS
																 ); // stride

			#pragma unroll
			for (int h = 0; h < HS/16; h++)
			{
				wmma_tf32_ldmatrix_m16n16k8<false/*is_A*/, 
																		true /*is_B_row_major*/>
																		(
																			&v_tile[c * 8][h * 16], 
														 				 	vks[0], vks[1], vks[2], vks[3],
																			HS // stride
																		);

				wmma_tf32_mma_m16n16k8<true /*is_B_row_major*/>(
					ps[0], ps[1], ps[2], ps[3],
					vks[0], vks[1], vks[2], vks[3],
					o_reg[h][0][0], o_reg[h][0][1], o_reg[h][1][0], o_reg[h][1][1],
					o_reg[h][0][2], o_reg[h][0][3], o_reg[h][1][2], o_reg[h][1][3]
				);
			}
		}

		__syncthreads();
	}

	// Normalize o_reg by l before writing back to global memory
	#pragma unroll
	for (int r = 0; r < 2; r++)
	{
		float l_inv = 1.0f / l[r];

		#pragma unroll
		for (int h = 0; h < HS/16; h++)
		{
			o_reg[h][r][0] *= l_inv;
			o_reg[h][r][1] *= l_inv;
			o_reg[h][r][2] *= l_inv;
			o_reg[h][r][3] *= l_inv;
		}
	}

	// Write back to global memory
	#pragma unroll
	for (int h = 0; h < HS/16; h++)
	{
		Reg32* o_reg_u32 = reinterpret_cast<Reg32*>(&o_reg[h][0][0]);

		wmma_f32_stmatrix_gmem_m16n16k8(
			&out[0],
			o_reg_u32[0], o_reg_u32[1], o_reg_u32[4], o_reg_u32[5],
			o_reg_u32[2], o_reg_u32[3], o_reg_u32[6], o_reg_u32[7],
			HS // stride
		);

		out += 16; // move to the next 16 columns
	}
}

void flash_attention(
	float* __restrict__ out,
	const float* __restrict__ q,
	const float* __restrict__ k,
	const float* __restrict__ v,
	const int B, const int T, const int NH, const int HS
) 
{
	constexpr int Bc = 64;
	constexpr int Wr = 16; // a warp handles 16 rows
	constexpr int Br = 64;
	constexpr int d = 64; // hardcoded head dimension

	assert(HS == d); // head dimension must be equal to d
	assert(T % Br == 0); // split along the sequence dimension

	constexpr int NUM_WARPS = Br / Wr; 
	constexpr int BLOCK_SIZE = NUM_WARPS * 32;
	dim3 grid(B * NH, T / Br);

	float scale = 1.0 / sqrtf(HS);
	flash_attention_kernel<Bc, Br, Wr, d, BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(out, q, k, v, T, scale);
}

void flash_attention_forward(
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
	// o is PV after value projection
	float* o = const_cast<float*>(inp);
	flash_attention(o, q, k, v, B, T, NH, HS);
	nvtxRangePop();

	nvtxRangePushA("unpermute_out");
	// unpermute from vac[B][NH][N][d] to out[B][N][NH][d]
	unpermute(o, out, B, T, NH, HS);
	nvtxRangePop();
}