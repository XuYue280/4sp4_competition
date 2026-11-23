// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "gemv.h"
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>


namespace swiftware::hpp {
    //TODO: Add more test

// Test basic gemv
TEST(GemvTest, BasicTest) {
    int m = 4, n = 4;
    float A[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float x[4] = {1,2,3,4};
    float y[4] = {0};
    
    ScheduleParams sp(2, 2);
    gemv(m, n, A, x, y, sp);
    
    EXPECT_NEAR(y[0], 1.0f, 1e-5);
    EXPECT_NEAR(y[1], 2.0f, 1e-5);
    EXPECT_NEAR(y[2], 3.0f, 1e-5);
    EXPECT_NEAR(y[3], 4.0f, 1e-5);
}

#ifdef USE_MKL
// Test gemv against MKL on a random matrix (small integer values, so the sums are exact)
TEST(GemvTest, RandomTest) {
    int m = 1000, n = 500;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-2, 2);
    std::vector<float> A(m * n), x(n), y(m), y_ref(m);
    for (auto &v : A) v = dist(gen);
    for (auto &v : x) v = dist(gen);
    for (int i = 0; i < m; ++i) y[i] = y_ref[i] = dist(gen);

    ScheduleParams sp(-1, -1);
    gemv(m, n, A.data(), x.data(), y.data(), sp);
    gemvMKL(m, n, A.data(), x.data(), y_ref.data(), sp);

    for (int i = 0; i < m; ++i) {
        ASSERT_NEAR(y[i], y_ref[i], 1e-5) << "at index " << i;
    }
}
#endif



}

