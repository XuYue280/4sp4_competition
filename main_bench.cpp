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

static void BM_GEMM(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    int t1 = state.range(3);
    int t2 = state.range(4);
    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);
    for (int i = 0; i < m * k; ++i) {
        A->data[i] = 1.0;
    }
    for (int i = 0; i < k * n; ++i) {
        B->data[i] = 1.0;
    }

    for (auto _: state) {
        swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }
    delete A;
    delete B;
    delete C;

}

static void BM_GEMV(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int t1 = state.range(2);
    int t2 = state.range(3);
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
        swiftware::hpp::gemv(m, n, A->data.data(), x->data.data(), y->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }
    delete A;
    delete x;
    delete y;
}

static void BM_SPMM(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    int t1 = state.range(3);
    int t2 = state.range(4);
    std::vector<int> Ap(m + 1);
    std::vector<int> Ai(m);
    std::vector<float> Ax(m, 1.0f);
    for (int i = 0; i < m; ++i) {
        Ap[i] = i;
        Ai[i] = i % k;
    }
    Ap[m] = m;
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);
    for (int i = 0; i < k * n; ++i) {
        B->data[i] = 1.0f;
    }

    for (auto _: state) {
        swiftware::hpp::spmmCSR(m, n, k, Ap.data(), Ai.data(), Ax.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }
    delete B;
    delete C;
}

static void BM_SPMV(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int t1 = state.range(2);
    int t2 = state.range(3);
    std::vector<int> Ap(m + 1);
    std::vector<int> Ai(m);
    std::vector<float> Ax(m, 1.0f);
    for (int i = 0; i < m; ++i) {
        Ap[i] = i;
        Ai[i] = i % n;
    }
    Ap[m] = m;
    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);
    for (int i = 0; i < n; ++i) {
        x->data[i] = 1.0f;
    }

    for (auto _: state) {
        swiftware::hpp::spmvCSR(m, n, Ap.data(), Ai.data(), Ax.data(), x->data.data(), y->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }
    delete x;
    delete y;
}

// For baseline and simd where tile sizes are not used
BENCHMARK(BM_GEMM)
    ->Args({4096, 4096, 4096, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_GEMV)
    ->Args({4096, 4096, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_SPMM)
    ->Args({4096, 4096, 4096, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);

BENCHMARK(BM_SPMV)
    ->Args({4096, 4096, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);




BENCHMARK_MAIN();
