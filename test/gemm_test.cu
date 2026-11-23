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

// Test the GEMM kernel against cuBLAS on random matrices (small integer values, so the sums are exact)
TEST(GemmGpuTest, RandomTest) {
    int m = 1000, n = 500, k = 100;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-2, 2);
    std::vector<float> A(m * k), B(k * n), C(m * n), C_ref(m * n);
    for (auto &v : A) v = dist(gen);
    for (auto &v : B) v = dist(gen);
    for (auto &v : C) v = dist(gen);

    float *d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    cudaMalloc(&d_C_ref, m * n * sizeof(float));
    cudaMemcpy(d_A, A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ref, C.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Reference: cuBLAS, C_ref = A * B + C_ref (column-major, so compute C^T = B^T * A^T)
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 1.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C_ref, n);
    cublasDestroy(handle);

    // TODO: Launch your GEMM kernel here, with the same configuration as in main_bench.cu
    // ...
    cudaError_t err = cudaDeviceSynchronize();

    cudaMemcpy(C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_ref.data(), d_C_ref, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);

    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    for (int i = 0; i < m * n; ++i) {
        ASSERT_NEAR(C[i], C_ref[i], 1e-5) << "at index " << i;
    }
}

}
