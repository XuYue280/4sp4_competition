// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmv.h"
#include <gtest/gtest.h>
#include <cmath>

namespace swiftware::hpp {
    //TODO: Add more tests

    // Test basic spmv with identity-like sparse matrix
    TEST(SpmvTest, BasicTest) {
        int m = 4, n = 4;
        // Create a sparse matrix in CSR format (diagonal matrix)
        int Ap[5] = {0, 1, 2, 3, 4};  // row pointers
        int Ai[4] = {0, 1, 2, 3};     // column indices
        float Ax[4] = {1.0f, 1.0f, 1.0f, 1.0f};  // values

        float b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float c[4] = {0.0f};

        ScheduleParams sp(-1, -1);
        spmvCSR(m, n, Ap, Ai, Ax, b, c, sp);

        EXPECT_NEAR(c[0], 1.0f, 1e-5);
        EXPECT_NEAR(c[1], 2.0f, 1e-5);
        EXPECT_NEAR(c[2], 3.0f, 1e-5);
        EXPECT_NEAR(c[3], 4.0f, 1e-5);
    }

}