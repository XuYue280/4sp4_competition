// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmm.h"
#include "gemm.h"
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

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

#ifdef USE_MKL
    // Test spmm against dense MKL gemm on a random 70% sparse matrix (small integer values)
    TEST(SpmmTest, RandomTest) {
        int m = 1000, n = 500, k = 100, sparsity = 70;
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(-2, 2);
        std::uniform_int_distribution<int> percent(0, 99);

        // Random sparse A, stored both as dense (for MKL) and as CSR
        std::vector<float> A(m * k, 0.0f);
        std::vector<int> Ap(m + 1, 0), Ai;
        std::vector<float> Ax;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k; ++j) {
                if (percent(gen) >= sparsity) {
                    A[i * k + j] = dist(gen);
                    Ai.push_back(j);
                    Ax.push_back(A[i * k + j]);
                }
            }
            Ap[i + 1] = Ai.size();
        }

        std::vector<float> B(k * n), C(m * n), C_ref(m * n);
        for (auto &v : B) v = dist(gen);
        for (int i = 0; i < m * n; ++i) C[i] = C_ref[i] = dist(gen);

        ScheduleParams sp(-1, -1);
        spmmCSR(m, n, k, Ap.data(), Ai.data(), Ax.data(), B.data(), C.data(), sp);
        gemmMKL(m, n, k, A.data(), B.data(), C_ref.data(), sp);

        for (int i = 0; i < m * n; ++i) {
            ASSERT_NEAR(C[i], C_ref[i], 1e-5) << "at index " << i;
        }
    }
#endif

}