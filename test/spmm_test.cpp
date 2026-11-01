// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmm.h"
#include <gtest/gtest.h>
#include <cmath>

namespace swiftware::hpp {
    //TODO: Add more tests

    // Test basic spmm with identity-like sparse matrix
    TEST(SpmmTest, BasicTest) {
        int m = 4, n = 4, k = 4;
        // Create a sparse matrix in CSR format (diagonal matrix)
        int Ap[5] = {0, 1, 2, 3, 4};  // row pointers
        int Ai[4] = {0, 1, 2, 3};     // column indices
        float Ax[4] = {1.0f, 1.0f, 1.0f, 1.0f};  // values

        // Dense matrix B (identity matrix)
        float B[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        float C[16] = {0};

        ScheduleParams sp(-1, -1);
        spmmCSR(m, n, k, Ap, Ai, Ax, B, C, sp);

        // Result should be close to the sparse matrix itself
        for (int i = 0; i < 16; ++i) {
            if (i % 5 == 0) {  // diagonal
                EXPECT_NEAR(C[i], 1.0f, 1e-5);
            } else {
                EXPECT_NEAR(C[i], 0.0f, 1e-5);
            }
        }
    }

}