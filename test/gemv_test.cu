// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "kernels.cuh"
#include <gtest/gtest.h>
#include <cublas_v2.h>
#include <random>
#include <vector>

namespace swiftware::hpp {

// Test the GEMV kernel against cuBLAS on a random matrix (small integer values, so the sums are exact)
TEST(GemvGpuTest, RandomTest) {
    int m = 1000, n = 500;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-2, 2);
    std::vector<float> A(m * n), x(n), y(m), y_ref(m);
    for (auto &v : A) v = dist(gen);
    for (auto &v : x) v = dist(gen);
    for (auto &v : y) v = dist(gen);

    float *d_A, *d_x, *d_y, *d_y_ref;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));
    cudaMalloc(&d_y_ref, m * sizeof(float));
    cudaMemcpy(d_A, A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_ref, y.data(), m * sizeof(float), cudaMemcpyHostToDevice);

    // Reference: cuBLAS, y_ref = A * x + y_ref (column-major, so the row-major A is used transposed)
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 1.0f;
    cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, d_A, n, d_x, 1, &beta, d_y_ref, 1);
    cublasDestroy(handle);

    // TODO: Launch your GEMV kernel here, with the same configuration as in main_bench.cu
    // ...
    cudaError_t err = cudaDeviceSynchronize();

    cudaMemcpy(y.data(), d_y, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y_ref.data(), d_y_ref, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_y_ref);

    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    for (int i = 0; i < m; ++i) {
        ASSERT_NEAR(y[i], y_ref[i], 1e-5) << "at index " << i;
    }
}

}
