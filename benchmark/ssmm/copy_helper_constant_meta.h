/*
 * Copyright (c) 2025 Chenpeng Wu (cpwu_sjtu@sjtu.edu.cn), Qiqi Gu (qiqi.gu@sjtu.edu.cn). 
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef FLEXIBLE_SPMM_COPY_HELPER_H
#define FLEXIBLE_SPMM_COPY_HELPER_H

#include <cuda_fp16.h>

#include "../mm_utils/memcpy_util.h"
#include "data_structure_helper.h"

// ========================= cfragment加载函数 =========================
__device__ __forceinline__
void mv_cfrag_x_p1(C_fragment<Shape_16_32_8, half> &src, C_fragment<Shape_16_32_8, half> &dst) {
    dst.x_p1[0] = src.x_p1[0];
}

__device__ __forceinline__
void mv_cfrag_x_p2(C_fragment<Shape_16_32_8, half> &src, C_fragment<Shape_16_32_8, half> &dst) {
    dst.x_p2[0] = src.x_p2[0];
}

__device__ __forceinline__
void mv_cfrag_x_p1(C_fragment<Shape_16_32_8, float> &src, C_fragment<Shape_16_32_8, float> &dst) {
    dst.x_p1[0] = src.x_p1[0];
    dst.x_p1[1] = src.x_p1[1];
}

__device__ __forceinline__
void mv_cfrag_x_p2(C_fragment<Shape_16_32_8, float> &src, C_fragment<Shape_16_32_8, float> &dst) {
    dst.x_p2[0] = src.x_p2[0];
    dst.x_p2[1] = src.x_p2[1];
}

// ========================= 数据类型转换函数 =========================

__device__ __forceinline__
half2 float2_to_half2(const float2 v) {
    return __floats2half2_rn(v.x, v.y);
}

__device__ __forceinline__
float2 half2_to_float2(const half2 v) {
    return __half22float2(v);
}

// ========================= EXACT: 1:4 to 2:4 expansion helper =========================

/**
 * Load 1:4 sparse data from global memory and expand to 2:4 format in shared memory
 * by duplicating each value. This enables true 1:4 storage while maintaining 2:4 hardware execution.
 * 
 * @param smem_tile Destination in shared memory (sized for 2:4 output)
 * @param gmem_panel Source in global memory (1:4 compact format)
 * @param values_num_cols_1_4 Number of columns in 1:4 format (k/4)
 * @param fetch Current tile index along K dimension
 * @param Block_M, Block_K Tile dimensions
 * @param threadIdx Thread index for parallel loading
 * @param aSwizzle Shared memory swizzle function
 */
template<typename ASwizzle, int A_GM_TO_SM_CP_SIZE, int A_GM_TO_SM_CP_ROWS_PER_ITER, int TOTAL_THREADS_PER_BLOCK>
__device__ __forceinline__
void load_and_expand_1_4_to_2_4(half* smem_tile, const half* gmem_panel,
                                 const int values_num_cols_1_4, const int fetch,
                                 const int Block_M, const int Block_K,
                                 const int threadIdx_x, ASwizzle aSwizzle) {
    const int input_cols_1_4 = Block_K / 4;
    const int output_cols_2_4 = Block_K / 2;

    // if (threadIdx_x == 0 && blockIdx.x == 0 && fetch == 0) {
    //     printf("[EXPAND] input_cols_1_4=%d, output_cols_2_4=%d, Block_M=%d, values_num_cols_1_4=%d\n",
    //         input_cols_1_4, output_cols_2_4, Block_M, values_num_cols_1_4);
    // }
    
    for (int row = threadIdx_x; row < Block_M; row += TOTAL_THREADS_PER_BLOCK) {
        for (int col_1_4 = 0; col_1_4 < input_cols_1_4; col_1_4++) {
            // Read from 1:4 compact storage, expand to 2:4 format
            int src_idx = row * values_num_cols_1_4 + fetch * input_cols_1_4 + col_1_4;
            
            half val = gmem_panel[src_idx];
            
            // Write duplicated to 2:4 positions for hardware execution
            int dst_base = row * (Block_K / 2) + col_1_4 * 2;
            smem_tile[aSwizzle(dst_base + 0)] = val;
            smem_tile[aSwizzle(dst_base + 1)] = val;
            
            // if (threadIdx_x == 0 && blockIdx.x == 0 && fetch == 0 && row == 0 && col_1_4 == 0) {
            //     printf("[EXPAND DEBUG] gmem_panel base ptr offset, row=%d, values_num_cols_1_4=%d, fetch=%d, src_idx=%d, val=%f\n",
            //         row, values_num_cols_1_4, fetch, src_idx, __half2float(val));
            //     printf("[EXPAND DEBUG] Block_K=%d, Block_K/2=%d, A_tile_size=%d\n",
            //         Block_K, Block_K/2, Block_M * Block_K / 2);
            //     printf("[EXPAND DEBUG] A_panel ptr=%p, offset 0 val=%f, offset 1 val=%f\n",
            //         gmem_panel, __half2float(gmem_panel[0]), __half2float(gmem_panel[1]));
            // }
        }
    }
    __syncthreads();
}

// template<typename ASwizzle, int A_GM_TO_SM_CP_SIZE, int A_GM_TO_SM_CP_ROWS_PER_ITER, int TOTAL_THREADS_PER_BLOCK>
// __device__ __forceinline__
// void load_and_expand_1_4_to_2_4(half* smem_tile, const half* gmem_block_base,
//                                  const int values_num_cols_1_4, const int fetch,
//                                  const int Block_M, const int Block_K,
//                                  const int threadIdx_x, ASwizzle aSwizzle) {
//     const int input_cols_1_4 = Block_K / 4;
//     const int output_cols_2_4 = Block_K / 2;
    
//     for (int row = threadIdx_x; row < Block_M; row += TOTAL_THREADS_PER_BLOCK) {
//         for (int col_1_4 = 0; col_1_4 < input_cols_1_4; col_1_4++) {
//             // Read from 1:4 compact storage, expand to 2:4 format
//             int src_idx = row * values_num_cols_1_4 + fetch * input_cols_1_4 + col_1_4;
//             half val = gmem_block_base[src_idx];
            
//             // Write duplicated to 2:4 positions for hardware execution
//             int dst_base = row * output_cols_2_4 + col_1_4 * 2;
//             smem_tile[aSwizzle(dst_base + 0)] = val;
//             smem_tile[aSwizzle(dst_base + 1)] = val;
//         }
//     }
//     __syncthreads();
// }

// ========================= 下面是async的数据加载 =========================

template<int SizeInBytes>
__device__ __forceinline__
void cp_gm_to_sm_async_pred_zfill(void *smem_ptr, void const *gmem_ptr,
                         const bool pred_guard = true, const bool zfill = false) {
    unsigned smem_int_ptr = get_smem_ptr(smem_ptr);
    int src_in_bytes = (zfill ? 0 : SizeInBytes);

    asm volatile (
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, %0, 0;\n"
            "  @p cp.async.cg.shared.global [%1], [%2], %3, %4;\n"
            "}\n"::"r"((int) pred_guard),
    "r"(smem_int_ptr), "l"(gmem_ptr), "n"(SizeInBytes), "r"(src_in_bytes)
            );
}

template<int SizeInBytes>
__device__ __forceinline__
void cp_gm_to_sm_async_zfill(void *smem_ptr, void const *gmem_ptr,
                    const bool zfill = false) {
    unsigned smem_int_ptr = get_smem_ptr(smem_ptr);
    int src_in_bytes = (zfill ? 0 : SizeInBytes);

    asm volatile (
            "{\n"
            "  cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
            "}\n"::"r"(smem_int_ptr), "l"(gmem_ptr), "n"(SizeInBytes), "r"(src_in_bytes)
            );
}

// ========================= 下面是sync的数据加载 =========================

__device__ __forceinline__
void load_meta_gm_to_frag_sync_short(uint *__restrict__ dst,
                          const uint *base, const int offset, int is_even, int meta_cols) {
    // *((short *) dst) = 0;
    *(((short *) dst)) = *((short *) (base + offset) + is_even);
    *(((short *) dst) + 1) = *((short *) (base + offset + 8 * meta_cols)  + is_even);
}

__device__ __forceinline__
void load_meta_gm_to_frag_sync_uint(uint *__restrict__ dst,
                                     const uint *base, const int offset) {
    *((uint *) dst) = *((uint *) (base + offset));
}

template<typename SWIZZLE>
__device__ __forceinline__
void load_float_sm_to_frag_sync_with_swizzle(uint *__restrict__ dst,
                                        const half *base, const int offset) {
    SWIZZLE swizzle;
    *((float *) dst) = *((float *) (base + swizzle(offset)));
}

template<typename SWIZZLE>
__device__ __forceinline__
void load_matrix_sm_to_frag_sync_no_trans(B_fragment<Shape_16_32_8> &dst,
                                          const half *base, const int offset, const int ldm, const int lane_id) {
    SWIZZLE swizzle;
    const half *src = base + swizzle(offset + lane_id % 8 * ldm + lane_id / 8 * 8);
    unsigned smem_ptr = get_smem_ptr(src);
    uint *x = dst.x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
            : "r"(smem_ptr));
}

template<typename SWIZZLE>
__device__ __forceinline__
void load_matrix_sm_to_frag_sync_no_trans(A_fragment<Shape_16_32_8> &dst,
                                          const half *base, const int offset, const int ldm, const int lane_id) {
    SWIZZLE swizzle;
    const half *src = base + swizzle(offset + lane_id % 8 * ldm + lane_id / 16 * 8 * ldm + lane_id % 16 / 8 * 8);
    unsigned smem_ptr = get_smem_ptr(src);
    uint *x = dst.x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
            : "r"(smem_ptr));
}

template<typename SWIZZLE>
__device__ __forceinline__
void store_matrix_frag_to_sm(const C_fragment<Shape_16_32_8, float> &dst,
                           const half *base, const int offset, const int ldm) {
    SWIZZLE swizzle;
    *((half2 *)(base + swizzle(offset))) = float2_to_half2(*((float2 *) dst.x_p1));
    *((half2 *)(base + swizzle(offset + 16 * ldm))) = float2_to_half2(*((float2 *) dst.x_p2));
}

template<typename SWIZZLE>
__device__ __forceinline__
void store_matrix_frag_to_sm(const C_fragment<Shape_16_32_8, float> &dst,
                             const half *base, const int offset) {
    SWIZZLE swizzle;
    *((half2 *)(base + swizzle(offset))) = float2_to_half2(*((float2 *) dst.x_p1));
    *((half2 *)(base + swizzle(offset + 16))) = float2_to_half2(*((float2 *) dst.x_p2));
}

template<typename SWIZZLE>
__device__ __forceinline__
void store_matrix_frag_to_sm(const C_fragment<Shape_16_32_8, half> &dst,
                             const half *base, const int offset, const int ldm) {
    SWIZZLE swizzle;
    *((uint *)(base + swizzle(offset))) = *(dst.x_p1);
    *((uint *)(base + swizzle(offset + 16 * ldm))) = *(dst.x_p2);
}

template<typename SWIZZLE>
__device__ __forceinline__
void store_matrix_frag_to_sm(const C_fragment<Shape_16_32_8, half> &dst,
                             const half *base, const int offset, const bool silu= false) {
    SWIZZLE swizzle;

    if(silu){
        float x = __half2float(((half*)(dst.x_p1))[1]);
        ((half*)(dst.x_p1))[0] = __float2half(x / (1.0f + expf(-x)));

        x = __half2float(((half*)(dst.x_p1))[1]);
        ((half*)(dst.x_p1))[1] = __float2half(x / (1.0f + expf(-x)));

        x = __half2float(((half*)(dst.x_p2))[0]);
        ((half*)(dst.x_p2))[0] = __float2half(x / (1.0f + expf(-x)));

        x = __half2float(((half*)(dst.x_p2))[1]);
        ((half*)(dst.x_p2))[1] = __float2half(x / (1.0f + expf(-x)));
    }

    *((uint *)(base + swizzle(offset))) = *(dst.x_p1);
    *((uint *)(base + swizzle(offset + 16))) = *(dst.x_p2);
}

template<typename SWIZZLE>
__device__ __forceinline__
void store_matrix_frag_to_sm_weight_dot(const C_fragment<Shape_16_32_8, half> &dst,
                             const half *base, const int offset, const half weight) {
    SWIZZLE swizzle;

    ((half*)(dst.x_p1))[0] = ((half*)(dst.x_p1))[0] * weight;
    ((half*)(dst.x_p1))[1] = ((half*)(dst.x_p1))[1] * weight;

    ((half*)(dst.x_p2))[0] = ((half*)(dst.x_p2))[0] * weight;
    ((half*)(dst.x_p2))[1] = ((half*)(dst.x_p2))[1] * weight;

    *((uint *)(base + swizzle(offset))) = *(dst.x_p1);
    *((uint *)(base + swizzle(offset + 16))) = *(dst.x_p2);
}

// for sm-90 architecture
template<typename SWIZZLE>
__device__ __forceinline__
void store_matrix_frag_to_sm_sync_no_trans(C_fragment<Shape_16_32_8, half> &src1, C_fragment<Shape_16_32_8, half> &src2,
                                          const half *base, const int offset) {
    SWIZZLE swizzle;
    const half *dst = base + swizzle(offset);
    unsigned smem_ptr = get_smem_ptr(dst);
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
            : "=r"(smem_ptr)
            : "r"(src1.x_p1[0]), "r"(src1.x_p2[0]), "r"(src2.x_p1[0]), "r"(src2.x_p2[0]));
}

#endif //FLEXIBLE_SPMM_COPY_HELPER_H
