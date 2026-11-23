// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "kernels.cuh"
#include <gtest/gtest.h>
#include <cusparse.h>
#include <random>
#include <vector>

namespace swiftware::hpp {

    // Test the SpMM kernel against cuSPARSE on a random 70% sparse matrix (small integer values)
    TEST(SpmmGpuTest, RandomTest) {
        int m = 1000, n = 500, k = 100, sparsity = 70;
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(-2, 2);
        std::uniform_int_distribution<int> percent(0, 99);

        // Random sparse A (m x k) in CSR format
        std::vector<int> Ap(m + 1, 0), Ai;
        std::vector<float> Ax;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k; ++j) {
                if (percent(gen) >= sparsity) {
                    Ai.push_back(j);
                    Ax.push_back(dist(gen));
                }
            }
            Ap[i + 1] = Ai.size();
        }
        int nnz = Ai.size();

        std::vector<float> B(k * n), C(m * n), C_ref(m * n);
        for (auto &v : B) v = dist(gen);
        for (auto &v : C) v = dist(gen);

        int *d_Ap, *d_Ai;
        float *d_Ax, *d_B, *d_C, *d_C_ref;
        cudaMalloc(&d_Ap, (m + 1) * sizeof(int));
        cudaMalloc(&d_Ai, nnz * sizeof(int));
        cudaMalloc(&d_Ax, nnz * sizeof(float));
        cudaMalloc(&d_B, k * n * sizeof(float));
        cudaMalloc(&d_C, m * n * sizeof(float));
        cudaMalloc(&d_C_ref, m * n * sizeof(float));
        cudaMemcpy(d_Ap, Ap.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ai, Ai.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ax, Ax.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, C.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C_ref, C.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);

        // Reference: cuSPARSE, C_ref = A * B + C_ref
        cusparseHandle_t handle;
        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matB, matC;
        cusparseCreate(&handle);
        cusparseCreateCsr(&matA, m, k, nnz, d_Ap, d_Ai, d_Ax,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateDnMat(&matB, k, n, n, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        cusparseCreateDnMat(&matC, m, n, n, d_C_ref, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        const float alpha = 1.0f, beta = 1.0f;
        size_t buffer_size = 0;
        void *d_buffer = nullptr;
        cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size);
        cudaMalloc(&d_buffer, buffer_size);
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buffer);
        cudaFree(d_buffer);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnMat(matB);
        cusparseDestroyDnMat(matC);
        cusparseDestroy(handle);

        // TODO: Launch your SpMM kernel here, with the same configuration as in main_bench.cu
        // ...
        cudaError_t err = cudaDeviceSynchronize();

        cudaMemcpy(C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref.data(), d_C_ref, m * n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_Ap);
        cudaFree(d_Ai);
        cudaFree(d_Ax);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_C_ref);

        ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
        for (int i = 0; i < m * n; ++i) {
            ASSERT_NEAR(C[i], C_ref[i], 1e-5) << "at index " << i;
        }
    }

}
