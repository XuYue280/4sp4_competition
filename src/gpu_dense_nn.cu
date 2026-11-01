// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gpu_dense_nn.cuh"
#include "kernels.cuh"

namespace swiftware::hpp {

    DenseMatrix *dense_nn_gemm_gpu(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp, cudaStream_t stream) {
        // TODO: IMPLEMENT THIS
        return new DenseMatrix(InData->m, 1);
    }

    DenseMatrix *dense_nn_gemv_gpu(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp, cudaStream_t stream) {
        // TODO: IMPLEMENT THIS
        return new DenseMatrix(InData->m, 1);
    }

}
