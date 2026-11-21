// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include <nvbench/nvbench.cuh>
#include "kernels.cuh"
#include "gpu_utils.h"
#include <vector>
#include <cmath>

void report_summary(nvbench::state& state)
{
    state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
    state.get_summary("nv/cold/time/cpu/mean").set_string("hide", "");
    state.get_summary("nv/cold/time/cpu/stdev/relative").set_string("hide", "");
}

// ============================================================================
// GEMM BENCHMARKS - BASELINE (cuBLAS)
// ============================================================================

void nvbench_gemm_baseline(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = m;
    const int k = static_cast<int>(state.get_int64("k"));

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    // Initialize with dummy data
    std::vector<float> h_A(m * k, 1.0f);
    std::vector<float> h_B(k * n, 1.0f);
    std::vector<float> h_C(m * n, 0.0f);

    cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // TODO: Call cuBLAS gemm baseline implementation
        timer.stop();
    });

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    report_summary(state);
}

// ============================================================================
// GEMM BENCHMARKS - OPTIMIZED
// ============================================================================

void nvbench_gemm_optimized(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = m;
    const int k = static_cast<int>(state.get_int64("k"));

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    // Initialize with dummy data
    std::vector<float> h_A(m * k, 1.0f);
    std::vector<float> h_B(k * n, 1.0f);
    std::vector<float> h_C(m * n, 0.0f);

    cudaMemcpy(d_A, h_A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // TODO: Call optimized gemm implementation
        timer.stop();
    });

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    report_summary(state);
}

// ============================================================================
// GEMV BENCHMARKS - BASELINE (cuBLAS)
// ============================================================================

void nvbench_gemv_baseline(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = m;

    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));

    std::vector<float> h_A(m * n, 1.0f);
    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(m, 0.0f);

    cudaMemcpy(d_A, h_A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // TODO: Call cuBLAS gemv baseline implementation
        timer.stop();
    });

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    report_summary(state);
}

// ============================================================================
// GEMV BENCHMARKS - OPTIMIZED
// ============================================================================

void nvbench_gemv_optimized(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = m;

    float *d_A, *d_x, *d_y;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));

    std::vector<float> h_A(m * n, 1.0f);
    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(m, 0.0f);

    cudaMemcpy(d_A, h_A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // TODO: Call optimized gemv implementation
        timer.stop();
    });

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    report_summary(state);
}

// ============================================================================
// SPMM BENCHMARKS - BASELINE (cuSPARSE)
// ============================================================================

void nvbench_spmm_baseline(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = m;
    const int k = static_cast<int>(state.get_int64("k"));
    const int sparsity = static_cast<int>(state.get_int64("sparsity"));

    // Allocate device memory for sparse and dense matrices
    float *d_B, *d_C;
    int *d_Ap, *d_Ai;
    float *d_Ax;

    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    // Create sparse matrix in CSR format
    double sparsity_ratio = sparsity / 100.0;
    int nnz = static_cast<int>((1.0 - sparsity_ratio) * m * k);
    std::vector<int> h_Ap(m + 1);
    std::vector<int> h_Ai(nnz);
    std::vector<float> h_Ax(nnz, 1.0f);

    int nnz_per_row = static_cast<int>((1.0 - sparsity_ratio) * k);
    nnz_per_row = (nnz_per_row > 0) ? nnz_per_row : 1;

    int cur_nnz = 0;
    for (int i = 0; i < m; ++i) {
        h_Ap[i] = cur_nnz;
        for (int j = 0; j < nnz_per_row && cur_nnz < nnz; ++j, ++cur_nnz) {
            h_Ai[cur_nnz] = (j * k / nnz_per_row) % k;
        }
    }
    h_Ap[m] = cur_nnz;

    cudaMalloc(&d_Ap, (m + 1) * sizeof(int));
    cudaMalloc(&d_Ai, nnz * sizeof(int));
    cudaMalloc(&d_Ax, nnz * sizeof(float));

    std::vector<float> h_B(k * n, 1.0f);
    cudaMemcpy(d_Ap, h_Ap.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ai, h_Ai.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ax, h_Ax.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // TODO: Call cuSPARSE spmm baseline implementation
        timer.stop();
    });

    cudaFree(d_Ap);
    cudaFree(d_Ai);
    cudaFree(d_Ax);
    cudaFree(d_B);
    cudaFree(d_C);

    report_summary(state);
}

// ============================================================================
// SPMM BENCHMARKS - OPTIMIZED
// ============================================================================

void nvbench_spmm_optimized(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = m;
    const int k = static_cast<int>(state.get_int64("k"));
    const int sparsity = static_cast<int>(state.get_int64("sparsity"));

    float *d_B, *d_C;
    int *d_Ap, *d_Ai;
    float *d_Ax;

    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    double sparsity_ratio = sparsity / 100.0;
    int nnz = static_cast<int>((1.0 - sparsity_ratio) * m * k);
    std::vector<int> h_Ap(m + 1);
    std::vector<int> h_Ai(nnz);
    std::vector<float> h_Ax(nnz, 1.0f);

    int nnz_per_row = static_cast<int>((1.0 - sparsity_ratio) * k);
    nnz_per_row = (nnz_per_row > 0) ? nnz_per_row : 1;

    int cur_nnz = 0;
    for (int i = 0; i < m; ++i) {
        h_Ap[i] = cur_nnz;
        for (int j = 0; j < nnz_per_row && cur_nnz < nnz; ++j, ++cur_nnz) {
            h_Ai[cur_nnz] = (j * k / nnz_per_row) % k;
        }
    }
    h_Ap[m] = cur_nnz;

    cudaMalloc(&d_Ap, (m + 1) * sizeof(int));
    cudaMalloc(&d_Ai, nnz * sizeof(int));
    cudaMalloc(&d_Ax, nnz * sizeof(float));

    std::vector<float> h_B(k * n, 1.0f);
    cudaMemcpy(d_Ap, h_Ap.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ai, h_Ai.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ax, h_Ax.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // TODO: Call optimized spmm implementation
        timer.stop();
    });

    cudaFree(d_Ap);
    cudaFree(d_Ai);
    cudaFree(d_Ax);
    cudaFree(d_B);
    cudaFree(d_C);

    report_summary(state);
}

// ============================================================================
// SPMV BENCHMARKS - BASELINE (cuSPARSE)
// ============================================================================

void nvbench_spmv_baseline(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = m;
    const int sparsity = static_cast<int>(state.get_int64("sparsity"));

    float *d_x, *d_y;
    int *d_Ap, *d_Ai;
    float *d_Ax;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));

    double sparsity_ratio = sparsity / 100.0;
    int nnz = static_cast<int>((1.0 - sparsity_ratio) * m * n);
    std::vector<int> h_Ap(m + 1);
    std::vector<int> h_Ai(nnz);
    std::vector<float> h_Ax(nnz, 1.0f);

    int nnz_per_row = static_cast<int>((1.0 - sparsity_ratio) * n);
    nnz_per_row = (nnz_per_row > 0) ? nnz_per_row : 1;

    int cur_nnz = 0;
    for (int i = 0; i < m; ++i) {
        h_Ap[i] = cur_nnz;
        for (int j = 0; j < nnz_per_row && cur_nnz < nnz; ++j, ++cur_nnz) {
            h_Ai[cur_nnz] = (j * n / nnz_per_row) % n;
        }
    }
    h_Ap[m] = cur_nnz;

    cudaMalloc(&d_Ap, (m + 1) * sizeof(int));
    cudaMalloc(&d_Ai, nnz * sizeof(int));
    cudaMalloc(&d_Ax, nnz * sizeof(float));

    std::vector<float> h_x(n, 1.0f);
    cudaMemcpy(d_Ap, h_Ap.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ai, h_Ai.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ax, h_Ax.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // TODO: Call cuSPARSE spmv baseline implementation
        timer.stop();
    });

    cudaFree(d_Ap);
    cudaFree(d_Ai);
    cudaFree(d_Ax);
    cudaFree(d_x);
    cudaFree(d_y);

    report_summary(state);
}

// ============================================================================
// SPMV BENCHMARKS - OPTIMIZED
// ============================================================================

void nvbench_spmv_optimized(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = m;
    const int sparsity = static_cast<int>(state.get_int64("sparsity"));

    float *d_x, *d_y;
    int *d_Ap, *d_Ai;
    float *d_Ax;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, m * sizeof(float));

    double sparsity_ratio = sparsity / 100.0;
    int nnz = static_cast<int>((1.0 - sparsity_ratio) * m * n);
    std::vector<int> h_Ap(m + 1);
    std::vector<int> h_Ai(nnz);
    std::vector<float> h_Ax(nnz, 1.0f);

    int nnz_per_row = static_cast<int>((1.0 - sparsity_ratio) * n);
    nnz_per_row = (nnz_per_row > 0) ? nnz_per_row : 1;

    int cur_nnz = 0;
    for (int i = 0; i < m; ++i) {
        h_Ap[i] = cur_nnz;
        for (int j = 0; j < nnz_per_row && cur_nnz < nnz; ++j, ++cur_nnz) {
            h_Ai[cur_nnz] = (j * n / nnz_per_row) % n;
        }
    }
    h_Ap[m] = cur_nnz;

    cudaMalloc(&d_Ap, (m + 1) * sizeof(int));
    cudaMalloc(&d_Ai, nnz * sizeof(int));
    cudaMalloc(&d_Ax, nnz * sizeof(float));

    std::vector<float> h_x(n, 1.0f);
    cudaMemcpy(d_Ap, h_Ap.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ai, h_Ai.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ax, h_Ax.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // TODO: Call optimized spmv implementation
        timer.stop();
    });

    cudaFree(d_Ap);
    cudaFree(d_Ai);
    cudaFree(d_Ax);
    cudaFree(d_x);
    cudaFree(d_y);

    report_summary(state);
}

// ============================================================================
// BENCHMARK REGISTRATION
// ============================================================================

NVBENCH_BENCH(nvbench_gemm_baseline)
    .set_name("gemm_baseline")
    .add_int64_axis("m", {256, 512, 1024, 2048, 4096, 8192, 16384 /*, 32768 */})
    .add_int64_axis("k", {256, 512, 1024, 2048 /*, 4096, 8192, 16384, 32768 */});

NVBENCH_BENCH(nvbench_gemm_optimized)
    .set_name("gemm_optimized")
    .add_int64_axis("m", {256, 512, 1024, 2048, 4096, 8192, 16384 /*, 32768 */})
    .add_int64_axis("k", {256, 512, 1024, 2048 /*, 4096, 8192, 16384, 32768 */});

NVBENCH_BENCH(nvbench_gemv_baseline)
    .set_name("gemv_baseline")
    .add_int64_axis("m", {256, 512, 1024, 2048, 4096, 8192, 16384 /*, 32768 */});

NVBENCH_BENCH(nvbench_gemv_optimized)
    .set_name("gemv_optimized")
    .add_int64_axis("m", {256, 512, 1024, 2048, 4096, 8192, 16384 /*, 32768 */});

NVBENCH_BENCH(nvbench_spmm_baseline)
    .set_name("spmm_baseline")
    .add_int64_axis("m", {256, 512, 1024, 2048, 4096, 8192, 16384 /*, 32768 */})
    .add_int64_axis("k", {256, 512, 1024, 2048 /*, 4096, 8192, 16384, 32768 */})
    .add_int64_axis("sparsity", {50, 70, 80, 90});

NVBENCH_BENCH(nvbench_spmm_optimized)
    .set_name("spmm_optimized")
    .add_int64_axis("m", {256, 512, 1024, 2048, 4096, 8192, 16384 /*, 32768 */})
    .add_int64_axis("k", {256, 512, 1024, 2048 /*, 4096, 8192, 16384, 32768 */})
    .add_int64_axis("sparsity", {50, 70, 80, 90});

NVBENCH_BENCH(nvbench_spmv_baseline)
    .set_name("spmv_baseline")
    .add_int64_axis("m", {256, 512, 1024, 2048, 4096, 8192, 16384 /*, 32768 */})
    .add_int64_axis("sparsity", {50, 70, 80, 90});

NVBENCH_BENCH(nvbench_spmv_optimized)
    .set_name("spmv_optimized")
    .add_int64_axis("m", {256, 512, 1024, 2048, 4096, 8192, 16384 /*, 32768 */})
    .add_int64_axis("sparsity", {50, 70, 80, 90});
