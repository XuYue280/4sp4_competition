// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gemm.h"
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

namespace swiftware::hpp {
//TODO: Add more test

// Test basic gemm
TEST(GemmTest, BasicTest) {
    int m = 4, n = 4, k = 4;
    float A[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float B[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float C[16] = {0};
    
    ScheduleParams sp(2, 2);
    gemm(m, n, k, A, B, C, sp);
    
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(C[i], A[i], 1e-5);
    }
}

#ifdef USE_MKL
// Test gemm against MKL on random matrices (small integer values, so the sums are exact)
TEST(GemmTest, RandomTest) {
    int m = 1000, n = 500, k = 100;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-2, 2);
    std::vector<float> A(m * k), B(k * n), C(m * n), C_ref(m * n);
    for (auto &v : A) v = dist(gen);
    for (auto &v : B) v = dist(gen);
    for (int i = 0; i < m * n; ++i) C[i] = C_ref[i] = dist(gen);

    ScheduleParams sp(-1, -1);
    gemm(m, n, k, A.data(), B.data(), C.data(), sp);
    gemmMKL(m, n, k, A.data(), B.data(), C_ref.data(), sp);

    for (int i = 0; i < m * n; ++i) {
        ASSERT_NEAR(C[i], C_ref[i], 1e-5) << "at index " << i;
    }
}
#endif


}

