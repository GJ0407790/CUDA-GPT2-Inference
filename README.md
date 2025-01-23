# CUDA GPT2 Inference

## Introduction

[TODO:] Introduce GPT2.

![alt text](./images/gpt2/gpt2_flow.png)

The image above shows the flow of the inference step:
- `encoder`:
- `layernorm`:
- `matmul`:
- `attention`:
- `residual`:
- `gelu`:

All of the experiments are done using a single A40 Nvidia GPU.

## Baseline

We will use Karpathy's [llm.c](https://github.com/karpathy/llm.c) as the baseline of our optimizations. Specifically, `train_gpt2_fp32.cu` is the code that we ran as the baseline.

![alt text](./images/llmc/llmc_breakdown.png)

As we can see from the figure above, `matmul` has the highest latency of ....

## Residual

Residual can be treated as vector addition. We have experimented with a few approaches as shown below.

### Approach 1: [Naive Kernel](https://github.com/GJ0407790/CUDA-GPT2-Inference/blob/main/kernels/residual/1_residual_naive.cuh)

We start by implementing a simple vector addition kernel.

```cuda
__global__ void residual_forward_naive_kernel(float* out, float* inp1, float* inp2, int N) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;

    if (t < N) 
    {
      out[t] = inp1[t] + inp2[t];
    }
}

void residual_forward_naive(float* out, float* inp1, float* inp2, int N) {
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;
  residual_forward_naive_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
}
```

The above code result in latency of **13.24ms** which is very close to the baseline latency of **12.3ms**. In fact, the baseline implementation is very similar to the above implementation. 

### Approach 2: [Cache Hints](https://github.com/GJ0407790/CUDA-GPT2-Inference/blob/main/kernels/residual/2_residual_cache_hint.cuh)

One optimization that can be done on approach 1 is using [cache hints](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#store-functions-using-cache-hints). This is the exact impelmentation in the baseline.

```cuda
__global__ void residual_forward_cache_hint_kernel(float* out, const float* inp1, const float* inp2, int N) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;

    if (t < N) 
    {
      out[t] = __ldcs(&inp1[t]) + __ldcs(&inp2[t]);
    }
}
```

However, using this approach resulted in **13.23ms** which is still **1ms** slower than the baseline approach. We suspect this is cause by the unoptimized kernel that precedes residual kernel.

## Layernorm

![alt text](./images/layernorm/layernorm.png)

Layernorm can be visualize using the image above, where elements of a same row are normalized such that the final mean is 0 and final standard deviation is 1 before going through a transformation. 

### Approach 1: [Block Reduction](https://github.com/GJ0407790/CUDA-GPT2-Inference/blob/main/kernels/layernorm/1_layernorm_block.cuh)

Since we have to iterate through all elements of a row to find the mean and standard deviation, an intuitive approach is to assign a block to handle a single row.

Each thread handles 4 elements in a row. For the ease of visualization, the row below has only 16 elements and each block has only 4 threads (during execution, the row has 768 elements and block size is 192).

![alt text](./images/layernorm/layernorm_block.png)

This approach took **27.17ms**, which is **3 times** slower than the baseline approach.

### Approach 2: [Vectorized Block Reduction](https://github.com/GJ0407790/CUDA-GPT2-Inference/blob/main/kernels/layernorm/2_layernorm_block_vectorized.cuh)

One small optimization that can be done is [vectorized memory access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/). In this approach, each thread will load 4 consecutive elements using a single load instruction.

![alt text](./images/layernorm/layernorm_block_vectorized.png)

Looking at assembly for approach 1 and 2, we can see different instructions used for loads, e.g. `LDG.E.CI.128, LDG.E.CI`.

```sass
layernorm_forward_block_vectorized_kernel:
 LDG.E.CI.128 R4, [R4]

layernorm_forward_block_kernel:
 LDG.E.CI R19, [R16]
```

This approach took **26.46ms** which is around **2.6%** improvement compared to approach 1.

### Approach 3: [Warp Reduction](https://github.com/GJ0407790/CUDA-GPT2-Inference/blob/main/kernels/layernorm/3_layernorm_warp.cuh)

Approach 2 has 2 downsides:

1. Atomic contention on 2 shared variables.
2. `__syncthreads` needed to coordinate the shared variables.

We can overcome the downsides mentioned above by letting a single warp to handle a row and communicate using either [cooperative groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cooperative-groups) or [warp shuffle functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions). However, this means that each thread has to handle more elements in a row, in our case, from 4 elements per thread to 24 elements per thread. This will result in higher register pressure since the elements are stored locally as registers.

This approach took **5.1ms** which achieved **5.19x** speedup compared to approach 2. In addition, this is **1.8x** faster than the baseline approach.

![alt text](./images/layernorm/warp_comparison.png)

Looking at the warp states, observed that approach 3 (blue) stalls significantly less than approach 2 (green) in:
- **Stall Short Scoreboard:** Indicates the contention to shared memory.
- **Stall Barrier:** Approach 3 has no synchronizations.

However, approach 3 still suffers stalls from long scoreboard which indicates that memory latency from global memory is not well hidden. In other words, approach 3 is memory bound which can be seen in the graph below.

![alt text](./images/layernorm/warp_throughput.png)

## Fused Residual and Layernorm

If we look at the flow graph in [Introduction](#introduction), we can observe that layernorm kernels are precede with residual kernels, except for the first layernorm kernel. 
In addition, both residual and layernorm kernels are memory bound, we could potentially merge these 2 kernels into a single kernel. This technique is known as kernel fusion.

Kernel fusion has several advantages:
- Reduce the number of memory requests to global memory as shown in the image below.
- Reduce multiple kernel launch overheads into single kernel launch overhead.

![alt text](./images/fused/fused.png)

The implementation can be found in [fused_residual_layernorm.cuh](./kernels/fused_residual_layernorm/fused_residual_layernorm.cuh).

![alt text](./images/fused/fused_comparison.png)

Using this approach resulted in total **10.1ms** which is **2.13x** speedup compared to the sum of residual and layernorm in baseline implementation. Note that there is still 1 instance of layernorm (the first instance) in the fusion appraoch.

## Matmul

Matmul at its core is matrix multiplication. This [article](https://siboehm.com/articles/22/CUDA-MMM) is strongly recommended that shows how to optimize matrix multiplication,

## Attention

![alt text](./images/attention/attention_breakdown.png)
