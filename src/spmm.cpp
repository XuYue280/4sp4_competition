// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmm.h"
#include "immintrin.h"

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace swiftware::hpp {

  void spmmCSR(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp) {
    // TODO: IMPLEMENT THIS: C = A * B + C
  }

#ifdef USE_MKL
  void spmmMKL(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp) {
    // MKL sparse BLAS SpMM: C = alpha*A*B + beta*C, A is m x k in CSR format
    // Since C may have initial values, we use beta=1.0 to accumulate
    sparse_matrix_t A;
    mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO, m, k,
                            const_cast<int *>(Ap),        // row start
                            const_cast<int *>(Ap) + 1,    // row end
                            const_cast<int *>(Ai), const_cast<float *>(Ax));
    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                    1.0f,                    // alpha
                    A, descr,
                    SPARSE_LAYOUT_ROW_MAJOR,
                    B, n, n,                 // B is k x n
                    1.0f,                    // beta (accumulate into C)
                    C, n);                   // C is m x n
    mkl_sparse_destroy(A);
  }
#endif

}