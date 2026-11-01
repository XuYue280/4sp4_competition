// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "sparse_nn.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>

namespace swiftware::hpp {
    //TODO: Add more tests

    // Test sparse_nn_spmm with small matrices
    TEST(SparseNNTest, SpmmBasicTest) {
        int batchSize = 2;
        int inputSize = 4;
        int hiddenSize = 3;
        int outputSize = 2;

        auto *InData = new DenseMatrix(batchSize, inputSize);

        // Create sparse matrices in CSR format
        auto *W1 = new CSR(hiddenSize, inputSize, hiddenSize);
        auto *W2 = new CSR(outputSize, hiddenSize, outputSize);

        auto *B1 = new DenseMatrix(hiddenSize, 1);
        auto *B2 = new DenseMatrix(outputSize, 1);

        // Initialize with small values
        for (int i = 0; i < InData->m * InData->n; ++i) InData->data[i] = 0.1f;
        for (int i = 0; i < B1->m * B1->n; ++i) B1->data[i] = 0.0f;
        for (int i = 0; i < B2->m * B2->n; ++i) B2->data[i] = 0.0f;

        // Initialize sparse matrix W1 (diagonal-like)
        for (int i = 0; i < hiddenSize; ++i) {
            W1->row_ptr[i] = i;
            W1->col_idx[i] = i;
            W1->values[i] = 0.1f;
        }
        W1->row_ptr[hiddenSize] = hiddenSize;

        // Initialize sparse matrix W2 (diagonal-like)
        for (int i = 0; i < outputSize; ++i) {
            W2->row_ptr[i] = i;
            W2->col_idx[i] = i;
            W2->values[i] = 0.1f;
        }
        W2->row_ptr[outputSize] = outputSize;

        ScheduleParams sp(-1, -1);
        DenseMatrix *result = sparseNNSpmm(InData, W1, W2, B1, B2, sp);

        // Result should have shape (batchSize, 1)
        ASSERT_EQ(result->m, batchSize);
        ASSERT_EQ(result->n, 1);

        delete InData;
        delete W1;
        delete W2;
        delete B1;
        delete B2;
        delete result;
    }

}