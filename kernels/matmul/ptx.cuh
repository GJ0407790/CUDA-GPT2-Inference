#pragma once

#include <cuda_bf16.h>
#include <cstdint> // for uint32_t

using Reg64 = uint64_t;
using Reg32 = uint32_t;
using bf16 = __nv_bfloat16;

__device__ __forceinline__ void cvt_bf16x2_f32x2(float2 f2, Reg32& reg)
{
	asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                 : "=r"(reg)
                 : "f"(f2.y),  
                   "f"(f2.x)); // the first is in lower 16 bits
}

/*
 * Async copy ptx
 */

__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) 
{
  uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
  uint64_t gmem_addr = __cvta_generic_to_global(gmem_ptr);

  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
               :: "r"(smem_addr), "l"(gmem_addr) : "memory");
} 

__device__ __forceinline__ void cp_async_commit_group()
{
  asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_group0()
{
  asm volatile("cp.async.wait_group 0;");
}

__device__ __forceinline__ void cp_async_wait_group1()
{
  asm volatile("cp.async.wait_group 1;");
}

/*
 * Tensor core related ptx
 */

// WMMA
template<bool is_A,
         bool is_B_row_major = false>
__device__ __forceinline__ void wmma_tf32_ldmatrix_m16n16k8(float* src, Reg32& d0, Reg32& d1, Reg32& d2, Reg32& d3, Reg32 stride)
{
  Reg32 smem_src = __cvta_generic_to_shared(src);

  if constexpr (is_A)
  {
    asm volatile(
      "wmma.load.a.sync.aligned.row.m16n16k8.shared.tf32 {%0, %1, %2, %3}, [%4], %5;\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
      : "r"(smem_src), "r"(stride)
    );
  }
  else
  {
    if constexpr (is_B_row_major)
    {
      asm volatile(
        "wmma.load.b.sync.aligned.row.m16n16k8.shared.tf32 {%0, %1, %2, %3}, [%4], %5;\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(smem_src), "r"(stride)
      ); 
    }
    else
    {
      asm volatile(
        "wmma.load.b.sync.aligned.col.m16n16k8.shared.tf32 {%0, %1, %2, %3}, [%4], %5;\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(smem_src), "r"(stride)
      );
    }
  }
}

__device__ __forceinline__ void wmma_f32_stmatrix_smem_m16n16k8(
  float* dst, 
  Reg32 const& d0, Reg32 const& d1, Reg32 const& d2, Reg32 const& d3, 
  Reg32 const& d4, Reg32 const& d5, Reg32 const& d6, Reg32 const& d7, 
  Reg32 stride)
{
  Reg32 smem_dst = __cvta_generic_to_shared(dst);

  asm volatile(
    "wmma.store.d.sync.aligned.row.m16n16k8.shared::cta.f32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}, %9;\n"
    : 
    : "r"(smem_dst), "r"(d0), "r"(d1), "r"(d2), "r"(d3), "r"(d4), "r"(d5), "r"(d6), "r"(d7), "r"(stride)
    : "memory"
  );
}

__device__ __forceinline__ void wmma_f32_stmatrix_gmem_m16n16k8(
  float* dst, 
  Reg32 const& d0, Reg32 const& d1, Reg32 const& d2, Reg32 const& d3, 
  Reg32 const& d4, Reg32 const& d5, Reg32 const& d6, Reg32 const& d7, 
  Reg32 stride)
{
  Reg64 gmem_dst = __cvta_generic_to_global(dst);

  asm volatile(
    "wmma.store.d.sync.aligned.row.m16n16k8.global.f32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}, %9;\n"
    : 
    : "l"(gmem_dst), "r"(d0), "r"(d1), "r"(d2), "r"(d3), "r"(d4), "r"(d5), "r"(d6), "r"(d7), "r"(stride)
  );
}

template<bool is_B_row_major = false>
__device__ __forceinline__ void wmma_tf32_mma_m16n16k8(
  Reg32 const& a0, Reg32 const& a1, Reg32 const& a2, Reg32 const& a3,
  Reg32 const& b0, Reg32 const& b1, Reg32 const& b2, Reg32 const& b3,
  float& c0, float& c1, float& c2, float& c3,
  float& c4, float& c5, float& c6, float& c7
)
{
  if constexpr (is_B_row_major)
  {
    asm volatile(
      "wmma.mma.sync.aligned.m16n16k8.row.row.f32.tf32.tf32.f32 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
      "{%8,  %9,  %10, %11},"
      "{%12, %13, %14, %15},"
      "{%16, %17, %18, %19, %20, %21, %22, %23};\n"
      : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3),
        "=f"(c4), "=f"(c5), "=f"(c6), "=f"(c7)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
        "r"(b0),  "r"(b1),  "r"(b2),  "r"(b3),
        "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
        "f"(c4),  "f"(c5),  "f"(c6),  "f"(c7)
    );
  }
  else
  {
    asm volatile(
      "wmma.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
      "{%8,  %9,  %10, %11},"
      "{%12, %13, %14, %15},"
      "{%16, %17, %18, %19, %20, %21, %22, %23};\n"
      : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3),
        "=f"(c4), "=f"(c5), "=f"(c6), "=f"(c7)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
        "r"(b0),  "r"(b1),  "r"(b2),  "r"(b3),
        "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
        "f"(c4),  "f"(c5),  "f"(c6),  "f"(c7)
    );
  }
}

// MMA
template<const int BK>
__device__ void ldmatrix_16x16(float* smem, Reg32& d0, Reg32& d1, Reg32& d2, Reg32& d3)
{
	// Implementation to load a 16x16 matrix from shared memory into registers
	// similar to ldmatrix_m8n8x4 but with manual conversion
	// Input smem points to the first element of the 8x8 tile on the upper left
	float2* smem_f2 = reinterpret_cast<float2*>(smem);

	cvt_bf16x2_f32x2(smem_f2[0], d0);
	cvt_bf16x2_f32x2(smem_f2[4 * BK], d1); // 8*BK for float, hence 4*BK for float2
	cvt_bf16x2_f32x2(smem_f2[4], d2);
	cvt_bf16x2_f32x2(smem_f2[4 * BK + 4], d3);
}

template<bool transpose>
__device__ __forceinline__ void ldmatrix_m8n8x4(bf16* src, Reg32& d0, Reg32& d1, Reg32& d2, Reg32& d3) 
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


