#pragma once

constexpr int TILE_WIDTH = 16;

__global__ void matmul_shared_kernel(
	float* __restrict__ out, 
	const float* __restrict__ inp, 
	const float* __restrict__ weight, 
	const float* __restrict__ bias, 
	const int T, 
	const int C, 
	const int OC) 
{
	__shared__ float inp_tile[TILE_WIDTH][TILE_WIDTH];
  __shared__ float weight_tile[TILE_WIDTH][TILE_WIDTH];

  int b = blockIdx.z;
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  int row_start = blockIdx.y * TILE_WIDTH;
  int col_start = blockIdx.x * TILE_WIDTH;
  float val = 0;

  for (int phase = 0; phase < C; phase += TILE_WIDTH) {
    if ((row_start + ty) < T && (phase + tx) < C) {
        inp_tile[ty][tx] = inp[(b * T + row_start + ty) * C + phase + tx];
    } else {
        inp_tile[ty][tx] = 0.0f;
    }

    if ((col_start + ty) < OC && (phase + tx) < C) {
        weight_tile[ty][tx] = weight[(col_start + ty)*C + phase + tx];
    } else {
        weight_tile[ty][tx] = 0.0f;
    }
    __syncthreads();

    for (int i = 0; i < TILE_WIDTH; i++) {
        val += inp_tile[ty][i] * weight_tile[tx][i];
    }
    
    __syncthreads();
  }

  if ((row_start + ty) < T && (col_start + tx) < OC) 
  {
    if (bias != NULL)
    {
      val += bias[col_start + tx];
    }

    out[(b*T + row_start + ty) * OC + col_start + tx] = val;
  }
}

void matmul_forward_tiling(
	float* __restrict__ out, 
	const float* __restrict__ inp, 
	const float* __restrict__ weight, 
  const float* __restrict__ bias, 
	const int B, 
	const int T, 
	const int C, 
	const int OC) 
{
	int gridx = (OC - 1) / TILE_WIDTH + 1;
  int gridy = (T - 1) / TILE_WIDTH + 1;
  dim3 grid_dim(gridx, gridy, B);
  dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);

  matmul_shared_kernel<<<grid_dim, block_dim>>>(out, inp, weight, bias, T, C, OC);
}
