// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#ifndef PROJECT_SPARSE_MATMUL_SPARSE_NN_H
#define PROJECT_SPARSE_MATMUL_SPARSE_NN_H
#include "def.h"
#include "spmm.h"
#include "spmv.h"

namespace swiftware::hpp {

  // CSR sparse matrix format
  struct CSR {
    int m;           // number of rows
    int n;           // number of columns
    int nnz;         // number of non-zero elements
    std::vector<int> row_ptr;    // row pointers (size m+1)
    std::vector<int> col_idx;    // column indices (size nnz)
    std::vector<float> values;   // non-zero values (size nnz)

    CSR(int m, int n, int nnz) : m(m), n(n), nnz(nnz),
        row_ptr(m+1, 0), col_idx(nnz), values(nnz) {}
  };

  //TODO add necessary includes

  // please do not change below
  DenseMatrix *sparseNNSpmm(DenseMatrix *InData, CSR *W1, CSR *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);

  DenseMatrix *sparseNNSpmv(DenseMatrix *InData, CSR *W1, CSR *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);


}
#endif //PROJECT_SPARSE_MATMUL_SPARSE_NN_H
