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


## Baseline

We will use Karpathy's [llm.c](https://github.com/karpathy/llm.c) as the baseline of our optimizations. Specifically, `train_gpt2_fp32.cu` is the code that we ran as the baseline.

![alt text](./images/llmc/llmc_breakdown.png)

As we can see from the figure above, `matmul` has the highest latency of ....

## Residual


## Layernorm

## Matmul

## Attention