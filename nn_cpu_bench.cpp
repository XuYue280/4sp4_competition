// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "benchmark/benchmark.h"
#include "utils.h"
#include "dense_nn.h"
#include "sparse_nn.h"

#include <string>

namespace {

struct MnistSplit {
    swiftware::hpp::DenseMatrix *raw;
    swiftware::hpp::DenseMatrix *labels;
    swiftware::hpp::DenseMatrix *features;

    MnistSplit() {
        raw = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
        labels = new swiftware::hpp::DenseMatrix(raw->m, 1);
        features = new swiftware::hpp::DenseMatrix(raw->m, raw->n - 1);
        // TODO: IMPLEMENT THIS
        // column 0 is the label; the other columns are pixels divided by 255
    }

    ~MnistSplit() {
        delete raw;
        delete labels;
        delete features;
    }
};

}

static void BM_DENSENN_GEMM(benchmark::State &state) {
    MnistSplit mnist;
    auto *w1 = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
    auto *w2 = swiftware::hpp::readCSV("./data/model/weights_output.csv");
    auto *b1 = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *b2 = swiftware::hpp::readCSV("./data/model/biases_output.csv");
    swiftware::hpp::ScheduleParams sp(state.range(0), state.range(1));

    for (auto _: state) {
        auto *pred = swiftware::hpp::dense_nn_gemm(mnist.features, w1, w2, b1, b2, sp);
        state.PauseTiming();
        // TODO: IMPLEMENT THIS
        delete pred;
        state.ResumeTiming();
    }

    delete w1;
    delete w2;
    delete b1;
    delete b2;
}

static void BM_DENSENN_GEMV(benchmark::State &state) {
    MnistSplit mnist;
    auto *w1 = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
    auto *w2 = swiftware::hpp::readCSV("./data/model/weights_output.csv");
    auto *b1 = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *b2 = swiftware::hpp::readCSV("./data/model/biases_output.csv");
    swiftware::hpp::ScheduleParams sp(state.range(0), state.range(1));

    for (auto _: state) {
        auto *pred = swiftware::hpp::dense_nn_gemv(mnist.features, w1, w2, b1, b2, sp);
        state.PauseTiming();
        // TODO: IMPLEMENT THIS
        delete pred;
        state.ResumeTiming();
    }

    delete w1;
    delete w2;
    delete b1;
    delete b2;
}

static void BM_SPARSENN_SPMM(benchmark::State &state) {
    MnistSplit mnist;
    std::string prefix = "./data/model/" + std::to_string(state.range(0));
    auto *w1Dense = swiftware::hpp::readCSV(prefix + "_W1.csv");
    auto *w2Dense = swiftware::hpp::readCSV(prefix + "_W2.csv");
    auto *w1 = swiftware::hpp::denseToCSR(w1Dense);
    auto *w2 = swiftware::hpp::denseToCSR(w2Dense);
    auto *b1 = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *b2 = swiftware::hpp::readCSV("./data/model/biases_output.csv");
    swiftware::hpp::ScheduleParams sp(-1, -1);

    for (auto _: state) {
        auto *pred = swiftware::hpp::sparseNNSpmm(mnist.features, w1, w2, b1, b2, sp);
        state.PauseTiming();
        // TODO: IMPLEMENT THIS
        delete pred;
        state.ResumeTiming();
    }

    delete w1Dense;
    delete w2Dense;
    delete w1;
    delete w2;
    delete b1;
    delete b2;
}

static void BM_SPARSENN_SPMV(benchmark::State &state) {
    MnistSplit mnist;
    std::string prefix = "./data/model/" + std::to_string(state.range(0));
    auto *w1Dense = swiftware::hpp::readCSV(prefix + "_W1.csv");
    auto *w2Dense = swiftware::hpp::readCSV(prefix + "_W2.csv");
    auto *w1 = swiftware::hpp::denseToCSR(w1Dense);
    auto *w2 = swiftware::hpp::denseToCSR(w2Dense);
    auto *b1 = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *b2 = swiftware::hpp::readCSV("./data/model/biases_output.csv");
    swiftware::hpp::ScheduleParams sp(-1, -1);

    for (auto _: state) {
        auto *pred = swiftware::hpp::sparseNNSpmv(mnist.features, w1, w2, b1, b2, sp);
        state.PauseTiming();
        // TODO: IMPLEMENT THIS
        delete pred;
        state.ResumeTiming();
    }

    delete w1Dense;
    delete w2Dense;
    delete w1;
    delete w2;
    delete b1;
    delete b2;
}

#ifdef USE_MKL
static void BM_DENSENN_MKL_GEMM(benchmark::State &state) {
    MnistSplit mnist;
    auto *w1 = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
    auto *w2 = swiftware::hpp::readCSV("./data/model/weights_output.csv");
    auto *b1 = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *b2 = swiftware::hpp::readCSV("./data/model/biases_output.csv");
    swiftware::hpp::ScheduleParams sp(state.range(0), state.range(1));

    for (auto _: state) {
        auto *pred = swiftware::hpp::dense_nn_mkl_gemm(mnist.features, w1, w2, b1, b2, sp);
        state.PauseTiming();
        // TODO: IMPLEMENT THIS
        delete pred;
        state.ResumeTiming();
    }

    delete w1;
    delete w2;
    delete b1;
    delete b2;
}

static void BM_DENSENN_MKL_GEMV(benchmark::State &state) {
    MnistSplit mnist;
    auto *w1 = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
    auto *w2 = swiftware::hpp::readCSV("./data/model/weights_output.csv");
    auto *b1 = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *b2 = swiftware::hpp::readCSV("./data/model/biases_output.csv");
    swiftware::hpp::ScheduleParams sp(state.range(0), state.range(1));

    for (auto _: state) {
        auto *pred = swiftware::hpp::dense_nn_mkl_gemv(mnist.features, w1, w2, b1, b2, sp);
        state.PauseTiming();
        // TODO: IMPLEMENT THIS
        delete pred;
        state.ResumeTiming();
    }

    delete w1;
    delete w2;
    delete b1;
    delete b2;
}
#endif

BENCHMARK(BM_DENSENN_GEMM)
    ->Args({-1, -1})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(1);

BENCHMARK(BM_DENSENN_GEMV)
    ->Args({-1, -1})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(1);

BENCHMARK(BM_SPARSENN_SPMM)
    ->DenseRange(50, 95, 5)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(1);

BENCHMARK(BM_SPARSENN_SPMV)
    ->DenseRange(50, 95, 5)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(1);

#ifdef USE_MKL
BENCHMARK(BM_DENSENN_MKL_GEMM)
    ->Args({-1, -1})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(1);

BENCHMARK(BM_DENSENN_MKL_GEMV)
    ->Args({-1, -1})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(1);
#endif

BENCHMARK_MAIN();
