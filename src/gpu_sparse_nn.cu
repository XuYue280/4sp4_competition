// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gpu_sparse_nn.cuh"
#include "kernels.cuh"

namespace swiftware::hpp {

    DenseMatrix *sparse_nn_spmm_gpu(DenseMatrix *InData, CSR *W1, CSR *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp, cudaStream_t stream) {
        // TODO: IMPLEMENT THIS
        return new DenseMatrix(InData->m, 1);
    }

    DenseMatrix *sparse_nn_spmv_gpu(DenseMatrix *InData, CSR *W1, CSR *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp, cudaStream_t stream) {
        // TODO: IMPLEMENT THIS
        return new DenseMatrix(InData->m, 1);
    }

}
