// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmv.h"
#include "gemv.h"
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

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

#ifdef USE_MKL
    // Test spmv against dense MKL gemv on a random 70% sparse matrix (small integer values)
    TEST(SpmvTest, RandomTest) {
        int m = 1000, n = 500, sparsity = 70;
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(-2, 2);
        std::uniform_int_distribution<int> percent(0, 99);

        // Random sparse A, stored both as dense (for MKL) and as CSR
        std::vector<float> A(m * n, 0.0f);
        std::vector<int> Ap(m + 1, 0), Ai;
        std::vector<float> Ax;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (percent(gen) >= sparsity) {
                    A[i * n + j] = dist(gen);
                    Ai.push_back(j);
                    Ax.push_back(A[i * n + j]);
                }
            }
            Ap[i + 1] = Ai.size();
        }

        std::vector<float> b(n), c(m), c_ref(m);
        for (auto &v : b) v = dist(gen);
        for (int i = 0; i < m; ++i) c[i] = c_ref[i] = dist(gen);

        ScheduleParams sp(-1, -1);
        spmvCSR(m, n, Ap.data(), Ai.data(), Ax.data(), b.data(), c.data(), sp);
        gemvMKL(m, n, A.data(), b.data(), c_ref.data(), sp);

        for (int i = 0; i < m; ++i) {
            ASSERT_NEAR(c[i], c_ref[i], 1e-5) << "at index " << i;
        }
    }
#endif

}