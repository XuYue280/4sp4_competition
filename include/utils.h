// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.
#ifndef PROJECT_DENSE_MATMUL_UTILS_H
#define PROJECT_DENSE_MATMUL_UTILS_H

#include <string>
#include "def.h"

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

 // Do not change the following function signatures
 /// \brief Read a CSV file and store it in a DenseMatrix
 /// \param filename Path to the CSV file
 /// \param OutMat Pointer to the DenseMatrix to store the data
 /// \param removeFirstRow Whether to remove the first row of the CSV file
 DenseMatrix * readCSV(const std::string &filename, bool removeFirstRow = false);

 /// \brief Convert a dense matrix to CSR format
 /// \param dense The dense matrix to convert
 /// \return CSR sparse matrix
 CSR * denseToCSR(DenseMatrix *dense);

}

#endif //PROJECT_DENSE_MATMUL_UTILS_H
