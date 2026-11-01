// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef LAB01_GPU_DENSE_NN_CUH
#define LAB01_GPU_DENSE_NN_CUH

#include "def.h"
#include <cuda_runtime.h>

namespace swiftware::hpp
{

    DenseMatrix *dense_nn_gemm_gpu(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp, cudaStream_t stream = 0);

    DenseMatrix *dense_nn_gemv_gpu(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp, cudaStream_t stream = 0);

}
#endif //LAB01_GPU_DENSE_NN_CUH
