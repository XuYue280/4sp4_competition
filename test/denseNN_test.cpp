// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "dense_nn.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>


namespace swiftware::hpp {
 //TODO: Add more tests

// Test dense_nn_gemm with small matrices
TEST(DenseNNTest, GemmBasicTest) {
    int batchSize = 2;
    int inputSize = 4;
    int hiddenSize = 3;
    int outputSize = 2;

    auto *InData = new DenseMatrix(batchSize, inputSize);
    auto *W1 = new DenseMatrix(hiddenSize, inputSize);
    auto *W2 = new DenseMatrix(outputSize, hiddenSize);
    auto *B1 = new DenseMatrix(hiddenSize, 1);
    auto *B2 = new DenseMatrix(outputSize, 1);

    // Initialize with small values
    for (int i = 0; i < InData->m * InData->n; ++i) InData->data[i] = 0.1f;
    for (int i = 0; i < W1->m * W1->n; ++i) W1->data[i] = 0.1f;
    for (int i = 0; i < W2->m * W2->n; ++i) W2->data[i] = 0.1f;
    for (int i = 0; i < B1->m * B1->n; ++i) B1->data[i] = 0.0f;
    for (int i = 0; i < B2->m * B2->n; ++i) B2->data[i] = 0.0f;

    ScheduleParams sp(-1, -1);
    DenseMatrix *result = dense_nn_gemm(InData, W1, W2, B1, B2, sp);

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
