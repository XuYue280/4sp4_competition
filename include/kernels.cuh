// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef PROJECT_KERNELS_CUH
#define PROJECT_KERNELS_CUH

#include <cuda_runtime.h>

namespace swiftware::hpp
{

    __global__ void GEMM(const float* a, const float* b, float* result, int m, int n, int k);

    __global__ void GEMV(const float* a, const float* b, float* result, int m, int n);

    __global__ void SpMM(int *row_ptr, int *col_id, const float* a, const float* b, float* result, int m, int n, int k);

    __global__ void SpMV(int *row_ptr, int *col_id, const float* a, const float* b, float* result, int m, int n);

}
#endif //PROJECT_KERNELS_CUH
