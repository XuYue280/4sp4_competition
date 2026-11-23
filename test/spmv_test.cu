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

    // Test the SpMV kernel against cuSPARSE on a random 70% sparse matrix (small integer values)
    TEST(SpmvGpuTest, RandomTest) {
        int m = 1000, n = 500, sparsity = 70;
        std::mt19937 gen(42);
        std::uniform_int_distribution<int> dist(-2, 2);
        std::uniform_int_distribution<int> percent(0, 99);

        // Random sparse A (m x n) in CSR format
        std::vector<int> Ap(m + 1, 0), Ai;
        std::vector<float> Ax;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (percent(gen) >= sparsity) {
                    Ai.push_back(j);
                    Ax.push_back(dist(gen));
                }
            }
            Ap[i + 1] = Ai.size();
        }
        int nnz = Ai.size();

        std::vector<float> b(n), c(m), c_ref(m);
        for (auto &v : b) v = dist(gen);
        for (auto &v : c) v = dist(gen);

        int *d_Ap, *d_Ai;
        float *d_Ax, *d_b, *d_c, *d_c_ref;
        cudaMalloc(&d_Ap, (m + 1) * sizeof(int));
        cudaMalloc(&d_Ai, nnz * sizeof(int));
        cudaMalloc(&d_Ax, nnz * sizeof(float));
        cudaMalloc(&d_b, n * sizeof(float));
        cudaMalloc(&d_c, m * sizeof(float));
        cudaMalloc(&d_c_ref, m * sizeof(float));
        cudaMemcpy(d_Ap, Ap.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ai, Ai.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ax, Ax.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, c.data(), m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c_ref, c.data(), m * sizeof(float), cudaMemcpyHostToDevice);

        // Reference: cuSPARSE, c_ref = A * b + c_ref
        cusparseHandle_t handle;
        cusparseSpMatDescr_t matA;
        cusparseDnVecDescr_t vecB, vecC;
        cusparseCreate(&handle);
        cusparseCreateCsr(&matA, m, n, nnz, d_Ap, d_Ai, d_Ax,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateDnVec(&vecB, n, d_b, CUDA_R_32F);
        cusparseCreateDnVec(&vecC, m, d_c_ref, CUDA_R_32F);
        const float alpha = 1.0f, beta = 1.0f;
        size_t buffer_size = 0;
        void *d_buffer = nullptr;
        cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecB, &beta, vecC, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
        cudaMalloc(&d_buffer, buffer_size);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, vecB, &beta, vecC, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);
        cudaFree(d_buffer);
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecB);
        cusparseDestroyDnVec(vecC);
        cusparseDestroy(handle);

        // TODO: Launch your SpMV kernel here, with the same configuration as in main_bench.cu
        // ...
        cudaError_t err = cudaDeviceSynchronize();

        cudaMemcpy(c.data(), d_c, m * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(c_ref.data(), d_c_ref, m * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_Ap);
        cudaFree(d_Ai);
        cudaFree(d_Ax);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFree(d_c_ref);

        ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
        for (int i = 0; i < m; ++i) {
            ASSERT_NEAR(c[i], c_ref[i], 1e-5) << "at index " << i;
        }
    }

}
