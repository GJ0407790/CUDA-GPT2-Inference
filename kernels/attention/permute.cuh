#pragma once

__global__ void permute_kernel(
  float* __restrict__ q, 
  float* __restrict__ k, 
  float* __restrict__ v,
	const float* __restrict__ inp,
	int B, int N, int NH, int d) 
{
	// input[B][T][3][C] can be interpreted as input[B][T][3][NH][HS] since C = NH * HS. 
	// grid goal is to split input into queries[B][NH][T][HS], keys[B][NH][T][HS], values[B][NH][T][HS]
	// where the number of sequences T = N. where the head size HS = d. 
	
	// each thread will have a unique B, T, NH, HS index and fill in the corresponding cell in q,k,v
	int flattened_output_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int batch_index = flattened_output_idx / (N*NH*d);
	int sequence_index = (flattened_output_idx % (N*NH*d)) / (NH*d);
	int head_index = ((flattened_output_idx % (N*NH*d)) % (NH*d)) / (d);
	int dimension_index = ((flattened_output_idx % (N*NH*d)) % (NH*d)) % (d);

	if (batch_index < B && sequence_index < N && head_index < NH && dimension_index < d) {
		int query_element_index = batch_index*(N*3*NH*d) + sequence_index*(3*NH*d) + 0*(NH*d) + head_index*(d) + dimension_index;
		int key_element_index = batch_index*(N*3*NH*d) + sequence_index*(3*NH*d) + 1*(NH*d) + head_index*(d) + dimension_index;
		int value_element_index = batch_index*(N*3*NH*d) + sequence_index*(3*NH*d) + 2*(NH*d) + head_index*(d) + dimension_index;

		q[batch_index*(NH * N * d) + head_index*(N * d) + sequence_index*(d) + dimension_index] = inp[query_element_index];
		k[batch_index*(NH * N * d) + head_index*(N * d) + sequence_index*(d) + dimension_index] = inp[key_element_index];
		v[batch_index*(NH * N * d) + head_index*(N * d) + sequence_index*(d) + dimension_index] = inp[value_element_index];
	}
}

__global__ void unpermute_kernel(
  const float* __restrict__ inp, 
  float* __restrict__ out, 
  int B, int N, int NH, int d) 
{
	// input[B][NH][N][d] -> output[B][N][NH][d]
	// each thread will have a unique B, N, NH, d index and grab its corresponding cell from inp
	int flattened_output_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int batch_index = flattened_output_idx / (N*NH*d);
	int sequence_index = (flattened_output_idx % (N*NH*d)) / (NH*d);
	int head_index = ((flattened_output_idx % (N*NH*d)) % (NH*d)) / (d);
	int dimension_index = ((flattened_output_idx % (N*NH*d)) % (NH*d)) % (d);

	if (batch_index < B && sequence_index < N && head_index < NH && dimension_index < d)
  {
		out[batch_index*(N * NH * d) + sequence_index*(NH * d) + head_index*(d) + dimension_index] = inp[batch_index*(N * NH * d) + head_index*(N * d) + sequence_index*(d) + dimension_index];
	}
}

void permute(
  float* __restrict__ q, 
  float* __restrict__ k, 
  float* __restrict__ v,
	const float* __restrict__ inp,
	int B, int T, int NH, int HS) 
{
  constexpr int BLOCK_SIZE = 768;
  int perm_num_blocks = (((B * NH * T * HS) - 1) / BLOCK_SIZE) + 1;
	permute_kernel<<<perm_num_blocks, BLOCK_SIZE>>>(q, k, v, inp, B, T, NH, HS);
}

void unpermute(
  const float* __restrict__ vac, 
  float* __restrict__ out, 
  int B, int T, int NH, int HS) 
{
  constexpr int BLOCK_SIZE = 768;
  int unperm_num_blocks = (((B * NH * T * HS) - 1) / BLOCK_SIZE) + 1;
  unpermute_kernel<<<unperm_num_blocks, BLOCK_SIZE>>>(vac, out, B, T, NH, HS);
}

