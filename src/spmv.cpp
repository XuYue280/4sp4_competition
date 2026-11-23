// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmv.h"

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace swiftware::hpp {
  void spmvCSR(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp){
    // TODO: IMPLEMENT THIS: c = A * b + c
  }

#ifdef USE_MKL
  void spmvMKL(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp){
    // MKL sparse BLAS SpMV: c = alpha*A*b + beta*c, A is m x n in CSR format
    // Since c may have initial values, we use beta=1.0 to accumulate
    sparse_matrix_t A;
    mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, m, n,
                            const_cast<int *>(Ap),        // row start
                            const_cast<int *>(Ap) + 1,    // row end
                            const_cast<int *>(Ai), const_cast<float *>(Ax));
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                    1.0f,                    // alpha
                    A, descr,
                    b,
                    1.0f,                    // beta (accumulate into c)
                    c);
    mkl_sparse_destroy(A);
  }
#endif

}