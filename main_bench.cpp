// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "benchmark/benchmark.h"
#include "gemm.h"
#include "gemv.h"
#include "spmm.h"
#include "spmv.h"
#include <vector>
#include <cmath>

// ============================================================================
// GEMM BENCHMARKS
// ============================================================================

#ifdef USE_MKL
static void BM_GEMM_Baseline(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);

    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);

    for (int i = 0; i < m * k; ++i) {
        A->data[i] = 1.0f;
    }
    for (int i = 0; i < k * n; ++i) {
        B->data[i] = 1.0f;
    }

    for (auto _: state) {
        swiftware::hpp::gemmMKL(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(-1, -1));
    }

    delete A;
    delete B;
    delete C;
}

BENCHMARK(BM_GEMM_Baseline)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->Args({4096, 4096, 4096})
    ->Args({8192, 8192, 8192})
    ->Args({16384, 16384, 16384})
    ->Args({32768, 32768, 32768})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);
#endif

static void BM_GEMM_Optimized(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);

    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);

    for (int i = 0; i < m * k; ++i) {
        A->data[i] = 1.0f;
    }
    for (int i = 0; i < k * n; ++i) {
        B->data[i] = 1.0f;
    }

    for (auto _: state) {
        swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(-1, -1));
    }

    delete A;
    delete B;
    delete C;
}

BENCHMARK(BM_GEMM_Optimized)
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Args({1024, 1024, 1024})
    ->Args({2048, 2048, 2048})
    ->Args({4096, 4096, 4096})
    ->Args({8192, 8192, 8192})
    ->Args({16384, 16384, 16384})
    ->Args({32768, 32768, 32768})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ============================================================================
// GEMV BENCHMARKS
// ============================================================================

#ifdef USE_MKL
static void BM_GEMV_Baseline(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);

    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);

    for (int i = 0; i < m * n; ++i) {
        A->data[i] = 1.0f;
    }
    for (int i = 0; i < n; ++i) {
        x->data[i] = 1.0f;
    }

    for (auto _: state) {
        swiftware::hpp::gemvMKL(m, n, A->data.data(), x->data.data(), y->data.data(), swiftware::hpp::ScheduleParams(-1, -1));
    }

    delete A;
    delete x;
    delete y;
}

BENCHMARK(BM_GEMV_Baseline)
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({2048, 2048})
    ->Args({4096, 4096})
    ->Args({8192, 8192})
    ->Args({16384, 16384})
    ->Args({32768, 32768})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);
#endif

static void BM_GEMV_Optimized(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);

    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);

    for (int i = 0; i < m * n; ++i) {
        A->data[i] = 1.0f;
    }
    for (int i = 0; i < n; ++i) {
        x->data[i] = 1.0f;
    }

    for (auto _: state) {
        swiftware::hpp::gemv(m, n, A->data.data(), x->data.data(), y->data.data(), swiftware::hpp::ScheduleParams(-1, -1));
    }

    delete A;
    delete x;
    delete y;
}

BENCHMARK(BM_GEMV_Optimized)
    ->Args({256, 256})
    ->Args({512, 512})
    ->Args({1024, 1024})
    ->Args({2048, 2048})
    ->Args({4096, 4096})
    ->Args({8192, 8192})
    ->Args({16384, 16384})
    ->Args({32768, 32768})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

// ============================================================================
// SPMM BENCHMARKS (with varying sparsity: 50%, 70%, 80%, 90%)
// ============================================================================

#ifdef USE_MKL
static void BM_SPMM_Baseline(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    double sparsity = state.range(3) / 100.0; // sparsity as percentage

    // Create CSR format sparse matrix
    int nnz = static_cast<int>((1.0 - sparsity) * m * k);
    std::vector<int> Ap(m + 1);
    std::vector<int> Ai(nnz);
    std::vector<float> Ax(nnz, 1.0f);

    // Simple distribution: each row has (1-sparsity)*k non-zeros
    int nnz_per_row = static_cast<int>((1.0 - sparsity) * k);
    nnz_per_row = (nnz_per_row > 0) ? nnz_per_row : 1;

    int cur_nnz = 0;
    for (int i = 0; i < m; ++i) {
        Ap[i] = cur_nnz;
        for (int j = 0; j < nnz_per_row && cur_nnz < nnz; ++j, ++cur_nnz) {
            Ai[cur_nnz] = (j * k / nnz_per_row) % k;
        }
    }
    Ap[m] = cur_nnz;

    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);

    for (int i = 0; i < k * n; ++i) {
        B->data[i] = 1.0f;
    }

    for (auto _: state) {
        // MKL doesn't have a direct spmmCSR function, so we call the optimized version
        swiftware::hpp::spmmCSR(m, n, k, Ap.data(), Ai.data(), Ax.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(-1, -1));
    }

    delete B;
    delete C;
}

BENCHMARK(BM_SPMM_Baseline)
    ->Args({256, 256, 256, 50})
    ->Args({256, 256, 256, 70})
    ->Args({256, 256, 256, 80})
    ->Args({256, 256, 256, 90})
    ->Args({512, 512, 512, 50})
    ->Args({512, 512, 512, 70})
    ->Args({512, 512, 512, 80})
    ->Args({512, 512, 512, 90})
    ->Args({1024, 1024, 1024, 50})
    ->Args({1024, 1024, 1024, 70})
    ->Args({1024, 1024, 1024, 80})
    ->Args({1024, 1024, 1024, 90})
    ->Args({2048, 2048, 2048, 50})
    ->Args({2048, 2048, 2048, 70})
    ->Args({2048, 2048, 2048, 80})
    ->Args({2048, 2048, 2048, 90})
    ->Args({4096, 4096, 4096, 50})
    ->Args({4096, 4096, 4096, 70})
    ->Args({4096, 4096, 4096, 80})
    ->Args({4096, 4096, 4096, 90})
    ->Args({8192, 8192, 8192, 50})
    ->Args({8192, 8192, 8192, 70})
    ->Args({8192, 8192, 8192, 80})
    ->Args({8192, 8192, 8192, 90})
    ->Args({16384, 16384, 16384, 50})
    ->Args({16384, 16384, 16384, 70})
    ->Args({16384, 16384, 16384, 80})
    ->Args({16384, 16384, 16384, 90})
    ->Args({32768, 32768, 32768, 50})
    ->Args({32768, 32768, 32768, 70})
    ->Args({32768, 32768, 32768, 80})
    ->Args({32768, 32768, 32768, 90})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(3);
#endif

static void BM_SPMM_Optimized(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    double sparsity = state.range(3) / 100.0;

    int nnz = static_cast<int>((1.0 - sparsity) * m * k);
    std::vector<int> Ap(m + 1);
    std::vector<int> Ai(nnz);
    std::vector<float> Ax(nnz, 1.0f);

    int nnz_per_row = static_cast<int>((1.0 - sparsity) * k);
    nnz_per_row = (nnz_per_row > 0) ? nnz_per_row : 1;

    int cur_nnz = 0;
    for (int i = 0; i < m; ++i) {
        Ap[i] = cur_nnz;
        for (int j = 0; j < nnz_per_row && cur_nnz < nnz; ++j, ++cur_nnz) {
            Ai[cur_nnz] = (j * k / nnz_per_row) % k;
        }
    }
    Ap[m] = cur_nnz;

    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);

    for (int i = 0; i < k * n; ++i) {
        B->data[i] = 1.0f;
    }

    for (auto _: state) {
        swiftware::hpp::spmmCSR(m, n, k, Ap.data(), Ai.data(), Ax.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(-1, -1));
    }

    delete B;
    delete C;
}

BENCHMARK(BM_SPMM_Optimized)
    ->Args({256, 256, 256, 50})
    ->Args({256, 256, 256, 70})
    ->Args({256, 256, 256, 80})
    ->Args({256, 256, 256, 90})
    ->Args({512, 512, 512, 50})
    ->Args({512, 512, 512, 70})
    ->Args({512, 512, 512, 80})
    ->Args({512, 512, 512, 90})
    ->Args({1024, 1024, 1024, 50})
    ->Args({1024, 1024, 1024, 70})
    ->Args({1024, 1024, 1024, 80})
    ->Args({1024, 1024, 1024, 90})
    ->Args({2048, 2048, 2048, 50})
    ->Args({2048, 2048, 2048, 70})
    ->Args({2048, 2048, 2048, 80})
    ->Args({2048, 2048, 2048, 90})
    ->Args({4096, 4096, 4096, 50})
    ->Args({4096, 4096, 4096, 70})
    ->Args({4096, 4096, 4096, 80})
    ->Args({4096, 4096, 4096, 90})
    ->Args({8192, 8192, 8192, 50})
    ->Args({8192, 8192, 8192, 70})
    ->Args({8192, 8192, 8192, 80})
    ->Args({8192, 8192, 8192, 90})
    ->Args({16384, 16384, 16384, 50})
    ->Args({16384, 16384, 16384, 70})
    ->Args({16384, 16384, 16384, 80})
    ->Args({16384, 16384, 16384, 90})
    ->Args({32768, 32768, 32768, 50})
    ->Args({32768, 32768, 32768, 70})
    ->Args({32768, 32768, 32768, 80})
    ->Args({32768, 32768, 32768, 90})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(3);

// ============================================================================
// SPMV BENCHMARKS (with varying sparsity: 50%, 70%, 80%, 90%)
// ============================================================================

#ifdef USE_MKL
static void BM_SPMV_Baseline(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    double sparsity = state.range(2) / 100.0;

    int nnz = static_cast<int>((1.0 - sparsity) * m * n);
    std::vector<int> Ap(m + 1);
    std::vector<int> Ai(nnz);
    std::vector<float> Ax(nnz, 1.0f);

    int nnz_per_row = static_cast<int>((1.0 - sparsity) * n);
    nnz_per_row = (nnz_per_row > 0) ? nnz_per_row : 1;

    int cur_nnz = 0;
    for (int i = 0; i < m; ++i) {
        Ap[i] = cur_nnz;
        for (int j = 0; j < nnz_per_row && cur_nnz < nnz; ++j, ++cur_nnz) {
            Ai[cur_nnz] = (j * n / nnz_per_row) % n;
        }
    }
    Ap[m] = cur_nnz;

    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);

    for (int i = 0; i < n; ++i) {
        x->data[i] = 1.0f;
    }

    for (auto _: state) {
        swiftware::hpp::spmvCSR(m, n, Ap.data(), Ai.data(), Ax.data(), x->data.data(), y->data.data(), swiftware::hpp::ScheduleParams(-1, -1));
    }

    delete x;
    delete y;
}

BENCHMARK(BM_SPMV_Baseline)
    ->Args({256, 256, 50})
    ->Args({256, 256, 70})
    ->Args({256, 256, 80})
    ->Args({256, 256, 90})
    ->Args({512, 512, 50})
    ->Args({512, 512, 70})
    ->Args({512, 512, 80})
    ->Args({512, 512, 90})
    ->Args({1024, 1024, 50})
    ->Args({1024, 1024, 70})
    ->Args({1024, 1024, 80})
    ->Args({1024, 1024, 90})
    ->Args({2048, 2048, 50})
    ->Args({2048, 2048, 70})
    ->Args({2048, 2048, 80})
    ->Args({2048, 2048, 90})
    ->Args({4096, 4096, 50})
    ->Args({4096, 4096, 70})
    ->Args({4096, 4096, 80})
    ->Args({4096, 4096, 90})
    ->Args({8192, 8192, 50})
    ->Args({8192, 8192, 70})
    ->Args({8192, 8192, 80})
    ->Args({8192, 8192, 90})
    ->Args({16384, 16384, 50})
    ->Args({16384, 16384, 70})
    ->Args({16384, 16384, 80})
    ->Args({16384, 16384, 90})
    ->Args({32768, 32768, 50})
    ->Args({32768, 32768, 70})
    ->Args({32768, 32768, 80})
    ->Args({32768, 32768, 90})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);
#endif

static void BM_SPMV_Optimized(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    double sparsity = state.range(2) / 100.0;

    int nnz = static_cast<int>((1.0 - sparsity) * m * n);
    std::vector<int> Ap(m + 1);
    std::vector<int> Ai(nnz);
    std::vector<float> Ax(nnz, 1.0f);

    int nnz_per_row = static_cast<int>((1.0 - sparsity) * n);
    nnz_per_row = (nnz_per_row > 0) ? nnz_per_row : 1;

    int cur_nnz = 0;
    for (int i = 0; i < m; ++i) {
        Ap[i] = cur_nnz;
        for (int j = 0; j < nnz_per_row && cur_nnz < nnz; ++j, ++cur_nnz) {
            Ai[cur_nnz] = (j * n / nnz_per_row) % n;
        }
    }
    Ap[m] = cur_nnz;

    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);

    for (int i = 0; i < n; ++i) {
        x->data[i] = 1.0f;
    }

    for (auto _: state) {
        swiftware::hpp::spmvCSR(m, n, Ap.data(), Ai.data(), Ax.data(), x->data.data(), y->data.data(), swiftware::hpp::ScheduleParams(-1, -1));
    }

    delete x;
    delete y;
}

BENCHMARK(BM_SPMV_Optimized)
    ->Args({256, 256, 50})
    ->Args({256, 256, 70})
    ->Args({256, 256, 80})
    ->Args({256, 256, 90})
    ->Args({512, 512, 50})
    ->Args({512, 512, 70})
    ->Args({512, 512, 80})
    ->Args({512, 512, 90})
    ->Args({1024, 1024, 50})
    ->Args({1024, 1024, 70})
    ->Args({1024, 1024, 80})
    ->Args({1024, 1024, 90})
    ->Args({2048, 2048, 50})
    ->Args({2048, 2048, 70})
    ->Args({2048, 2048, 80})
    ->Args({2048, 2048, 90})
    ->Args({4096, 4096, 50})
    ->Args({4096, 4096, 70})
    ->Args({4096, 4096, 80})
    ->Args({4096, 4096, 90})
    ->Args({8192, 8192, 50})
    ->Args({8192, 8192, 70})
    ->Args({8192, 8192, 80})
    ->Args({8192, 8192, 90})
    ->Args({16384, 16384, 50})
    ->Args({16384, 16384, 70})
    ->Args({16384, 16384, 80})
    ->Args({16384, 16384, 90})
    ->Args({32768, 32768, 50})
    ->Args({32768, 32768, 70})
    ->Args({32768, 32768, 80})
    ->Args({32768, 32768, 90})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);

BENCHMARK_MAIN();
