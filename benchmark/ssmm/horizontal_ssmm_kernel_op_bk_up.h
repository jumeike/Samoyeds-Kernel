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


#ifndef FLEXIBLE_SPMM_HORIZONTAL_SSMM_KERNEL_OP_H
#define FLEXIBLE_SPMM_HORIZONTAL_SSMM_KERNEL_OP_H

#include "horizontal_ssmm_kernel.h"
#include "../util/utils.h"

template<typename KernelType>
__global__ void _HorizontalSsmmKernel(const int m, const int n, const int k,
                                      // const int V, const int N, const int M,
                                      const half *A_values, const uint *A_metadata, const uint *A_indices,
                                      const half *B, const uint *B_indices, const uint B_indices_len,
                                      const half *C, half *D) {
    KernelType kernel;
    extern __shared__ half shared_mem_workspace[];

    kernel.mainLoop(m, n, k, 0, 0, A_values, A_metadata, A_indices, B, B_indices, B_indices_len,
                    shared_mem_workspace);
    kernel.epilogue(m, n, B_indices_len, C, D, shared_mem_workspace);
}

template<typename KernelType>
__global__ void _HorizontalSsmmTransKernel(const int m, const int n, const int k,
                                      const half *A_values, const uint *A_metadata, const uint *A_indices,
                                      const half *B, const uint *B_indices, const uint B_indices_len,
                                      const half *C, half *D) {
    KernelType kernel;
    extern __shared__ half shared_mem_workspace[];

    kernel.mainLoopTrans(m, n, k, 0, 0, A_values, A_metadata, A_indices, B, B_indices, B_indices_len,
                    shared_mem_workspace, false);
    kernel.epilogueTrans(m, n, B_indices_len, D, shared_mem_workspace);
}

template<typename KernelType>
__global__ void _HorizontalSsmmFusedActTransKernel(const int m, const int n, const int k,
                                           const half *A_values, const uint *A_metadata, const uint *A_indices,
                                           const half *B, const uint *B_indices, const uint B_indices_len,
                                           const half *C, half *D) {
    KernelType kernel;
    extern __shared__ half shared_mem_workspace[];

    kernel.mainLoopTrans(m, n, k, 0, 0, A_values, A_metadata, A_indices, B, B_indices, B_indices_len,
                         shared_mem_workspace, true);
    kernel.epilogueTrans(m, n, B_indices_len, D, shared_mem_workspace);
}

// Helper template to dispatch based on N:M runtime values
template<typename BlockShape, typename WarpShape, typename MmaShape, int NStage, int N, int M>
void HorizontalSsmmKernelExecImpl(const int m, const int n, const int k,
                                   const half *A_values, const uint *A_metadata, const uint *A_indices,
                                   const half *B, const uint *B_indices, const uint B_indices_len,
                                   const half *C, half *D) {
    using AccumulatorType = half;
    using ASwizzle = Swizzle8BWiseXor;
    using BSwizzle = Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;

    using KernelType = HorizontalSsmmKernel<SparseRatioBase<N, M>, 128, BlockShape, WarpShape, MmaShape, NStage,
                AccumulatorType, ASwizzle, BSwizzle, CSwizzle>;
    size_t shared_mem_size = max(KernelType::input_buffer_size, KernelType::output_buffer_size);
    cudaStream_t stream = NULL;
    int dim_x = m / KernelType::Block_M / M * N;
    int dim_y = B_indices_len / KernelType::Block_N;
    _HorizontalSsmmKernel<KernelType><<<dim3(dim_x, dim_y, 1), dim3(128, 1, 1), shared_mem_size, stream>>>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
}

template<typename BlockShape, typename WarpShape, typename MmaShape,
        int NStage>
void HorizontalSsmmKernelExec(const int m, const int n, const int k,
                              const int V, const int N_param, const int M_param,
                              const half *A_values, const uint *A_metadata, const uint *A_indices,
                              const half *B, const uint *B_indices, const uint B_indices_len,
                              const half *C, half *D) {
    // Runtime dispatch based on N:M sparsity pattern
    if (N_param == 1 && M_param == 2) {
        HorizontalSsmmKernelExecImpl<BlockShape, WarpShape, MmaShape, NStage, 1, 2>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
    } else if (N_param == 2 && M_param == 4) {
        HorizontalSsmmKernelExecImpl<BlockShape, WarpShape, MmaShape, NStage, 2, 4>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
    } else if (N_param == 1 && M_param == 4) {
        HorizontalSsmmKernelExecImpl<BlockShape, WarpShape, MmaShape, NStage, 1, 4>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
    } else {
        printf("ERROR: Unsupported N:M sparsity pattern (%d:%d)\n", N_param, M_param);
    }
}

// Helper template for Trans variant
template<typename BlockShape, typename WarpShape, typename MmaShape, int NStage, int N, int M>
void HorizontalSsmmTransKernelExecImpl(const int m, const int n, const int k,
                                        const half *A_values, const uint *A_metadata, const uint *A_indices,
                                        const half *B, const uint *B_indices, const uint B_indices_len,
                                        const half *C, half *D) {
    using AccumulatorType = half;
    using ASwizzle = Swizzle8BWiseXor;
    using BSwizzle = Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;

    using KernelType = HorizontalSsmmKernel<SparseRatioBase<N, M>, 128, BlockShape, WarpShape, MmaShape, NStage,
            AccumulatorType, ASwizzle, BSwizzle, CSwizzle>;
    size_t shared_mem_size = max(KernelType::input_buffer_size, KernelType::output_buffer_size);
    cudaStream_t stream = NULL;
    int dim_x = m / KernelType::Block_M / M * N;
    int dim_y = B_indices_len / KernelType::Block_N;
    _HorizontalSsmmTransKernel<KernelType><<<dim3(dim_x, dim_y, 1), dim3(128, 1, 1), shared_mem_size, stream>>>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
}

template<typename BlockShape, typename WarpShape, typename MmaShape,
        int NStage>
void HorizontalSsmmTransKernelExec(const int m, const int n, const int k,
                              const int V, const int N_param, const int M_param,
                              const half *A_values, const uint *A_metadata, const uint *A_indices,
                              const half *B, const uint *B_indices, const uint B_indices_len,
                              const half *C, half *D) {
    // Runtime dispatch based on N:M sparsity pattern
    if (N_param == 1 && M_param == 2) {
        HorizontalSsmmTransKernelExecImpl<BlockShape, WarpShape, MmaShape, NStage, 1, 2>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
    } else if (N_param == 2 && M_param == 4) {
        HorizontalSsmmTransKernelExecImpl<BlockShape, WarpShape, MmaShape, NStage, 2, 4>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
    } else if (N_param == 1 && M_param == 4) {
        HorizontalSsmmTransKernelExecImpl<BlockShape, WarpShape, MmaShape, NStage, 1, 4>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
    } else {
        printf("ERROR: Unsupported N:M sparsity pattern (%d:%d) in Trans\n", N_param, M_param);
    }
}

// Helper template for FusedAct variant
template<typename BlockShape, typename WarpShape, typename MmaShape, int NStage, int N, int M>
void HorizontalSsmmFusedActTransKernelExecImpl(const int m, const int n, const int k,
                                                const half *A_values, const uint *A_metadata, const uint *A_indices,
                                                const half *B, const uint *B_indices, const uint B_indices_len,
                                                const half *C, half *D) {
    using AccumulatorType = half;
    using ASwizzle = Swizzle8BWiseXor;
    using BSwizzle = Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;

    using KernelType = HorizontalSsmmKernel<SparseRatioBase<N, M>, 128, BlockShape, WarpShape, MmaShape, NStage,
            AccumulatorType, ASwizzle, BSwizzle, CSwizzle>;
    size_t shared_mem_size = max(KernelType::input_buffer_size, KernelType::output_buffer_size);
    cudaStream_t stream = NULL;
    int dim_x = m / KernelType::Block_M / M * N;
    int dim_y = B_indices_len / KernelType::Block_N;
    _HorizontalSsmmFusedActTransKernel<KernelType><<<dim3(dim_x, dim_y, 1), dim3(128, 1, 1), shared_mem_size, stream>>>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
}

template<typename BlockShape, typename WarpShape, typename MmaShape,
        int NStage>
void HorizontalSsmmFusedActTransKernelExec(const int m, const int n, const int k,
                                   const int V, const int N_param, const int M_param,
                                   const half *A_values, const uint *A_metadata, const uint *A_indices,
                                   const half *B, const uint *B_indices, const uint B_indices_len,
                                   const half *C, half *D) {
    // Runtime dispatch based on N:M sparsity pattern
    if (N_param == 1 && M_param == 2) {
        HorizontalSsmmFusedActTransKernelExecImpl<BlockShape, WarpShape, MmaShape, NStage, 1, 2>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
    } else if (N_param == 2 && M_param == 4) {
        HorizontalSsmmFusedActTransKernelExecImpl<BlockShape, WarpShape, MmaShape, NStage, 2, 4>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
    } else if (N_param == 1 && M_param == 4) {
        HorizontalSsmmFusedActTransKernelExecImpl<BlockShape, WarpShape, MmaShape, NStage, 1, 4>(
            m, n, k, A_values, A_metadata, A_indices, B, B_indices, B_indices_len, C, D);
    } else {
        printf("ERROR: Unsupported N:M sparsity pattern (%d:%d) in FusedAct\n", N_param, M_param);
    }
}

#endif //FLEXIBLE_SPMM_HORIZONTAL_SSMM_KERNEL_OP_H
