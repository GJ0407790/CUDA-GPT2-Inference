#pragma once

#include <cuda_bf16.h>
#include <cstdint> // for uint32_t

using Reg32 = uint32_t;
using bf16 = __nv_bfloat16;

template<bool transpose>
__device__ __forceinline__ void ldmatrix_m16n16x4(bf16* src, Reg32& d0, Reg32& d1, Reg32& d2, Reg32& d3) 
{
  Reg32 smem_src = __cvta_generic_to_shared(src);
  
  if constexpr (transpose) {
    asm volatile(
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(smem_src)
    );
  } else {
    asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(smem_src)
    );
  }
}

__device__ __forceinline__ void mma_m16n8k16(
  Reg32 const& a0, Reg32 const& a1, Reg32 const& a2, Reg32 const& a3,
  Reg32 const& b0, Reg32 const& b1,
  float& c0, float& c1, float& c2, float& c3
)
{
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13};\n"
    : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
    :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
       "r"(b0),  "r"(b1),
       "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3)
  );
}


