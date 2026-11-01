// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "kernels.cuh"

namespace swiftware::hpp
{

    __global__ void GEMM(const float* a, const float* b, float* result, int m, int n, int k)
    {
        // TODO: IMPLEMENT THIS
    }

    __global__ void GEMV(const float* a, const float* b, float* result, int m, int n)
    {
        // TODO: IMPLEMENT THIS
    }

    __global__ void SpMM(int *row_ptr, int *col_id, const float* a, const float* b, float* result, int m, int n, int k)
    {
        // TODO: IMPLEMENT THIS
    }

    __global__ void SpMV(int *row_ptr, int *col_id, const float* a, const float* b, float* result, int m, int n)
    {
        // TODO: IMPLEMENT THIS
    }

}
