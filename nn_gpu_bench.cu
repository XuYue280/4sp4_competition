// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include <nvbench/nvbench.cuh>
#include <string>
#include "utils.h"
#include "gpu_dense_nn.cuh"
#include "gpu_sparse_nn.cuh"

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

void nvbench_dense_nn_gemm(nvbench::state &state) {
    MnistSplit mnist;
    auto *w1 = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
    auto *w2 = swiftware::hpp::readCSV("./data/model/weights_output.csv");
    auto *b1 = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *b2 = swiftware::hpp::readCSV("./data/model/biases_output.csv");
    swiftware::hpp::ScheduleParams sp(-1, -1);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
        timer.start();
        auto *pred = swiftware::hpp::dense_nn_gemm_gpu(mnist.features, w1, w2, b1, b2, sp, launch.get_stream());
        timer.stop();
        // TODO: IMPLEMENT THIS
        delete pred;
    });

    delete w1;
    delete w2;
    delete b1;
    delete b2;
}

void nvbench_dense_nn_gemv(nvbench::state &state) {
    MnistSplit mnist;
    auto *w1 = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
    auto *w2 = swiftware::hpp::readCSV("./data/model/weights_output.csv");
    auto *b1 = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *b2 = swiftware::hpp::readCSV("./data/model/biases_output.csv");
    swiftware::hpp::ScheduleParams sp(-1, -1);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
        timer.start();
        auto *pred = swiftware::hpp::dense_nn_gemv_gpu(mnist.features, w1, w2, b1, b2, sp, launch.get_stream());
        timer.stop();
        // TODO: IMPLEMENT THIS
        delete pred;
    });

    delete w1;
    delete w2;
    delete b1;
    delete b2;
}

void nvbench_sparse_nn_spmm(nvbench::state &state) {
    MnistSplit mnist;
    std::string prefix = "./data/model/" + std::to_string(state.get_int64("sparsity"));
    auto *w1Dense = swiftware::hpp::readCSV(prefix + "_W1.csv");
    auto *w2Dense = swiftware::hpp::readCSV(prefix + "_W2.csv");
    auto *w1 = swiftware::hpp::denseToCSR(w1Dense);
    auto *w2 = swiftware::hpp::denseToCSR(w2Dense);
    auto *b1 = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *b2 = swiftware::hpp::readCSV("./data/model/biases_output.csv");
    swiftware::hpp::ScheduleParams sp(-1, -1);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
        timer.start();
        auto *pred = swiftware::hpp::sparse_nn_spmm_gpu(mnist.features, w1, w2, b1, b2, sp, launch.get_stream());
        timer.stop();
        // TODO: IMPLEMENT THIS
        delete pred;
    });

    delete w1Dense;
    delete w2Dense;
    delete w1;
    delete w2;
    delete b1;
    delete b2;
}

void nvbench_sparse_nn_spmv(nvbench::state &state) {
    MnistSplit mnist;
    std::string prefix = "./data/model/" + std::to_string(state.get_int64("sparsity"));
    auto *w1Dense = swiftware::hpp::readCSV(prefix + "_W1.csv");
    auto *w2Dense = swiftware::hpp::readCSV(prefix + "_W2.csv");
    auto *w1 = swiftware::hpp::denseToCSR(w1Dense);
    auto *w2 = swiftware::hpp::denseToCSR(w2Dense);
    auto *b1 = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *b2 = swiftware::hpp::readCSV("./data/model/biases_output.csv");
    swiftware::hpp::ScheduleParams sp(-1, -1);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
        timer.start();
        auto *pred = swiftware::hpp::sparse_nn_spmv_gpu(mnist.features, w1, w2, b1, b2, sp, launch.get_stream());
        timer.stop();
        // TODO: IMPLEMENT THIS
        delete pred;
    });

    delete w1Dense;
    delete w2Dense;
    delete w1;
    delete w2;
    delete b1;
    delete b2;
}

NVBENCH_BENCH(nvbench_dense_nn_gemm).set_name("dense_nn_gemm");
NVBENCH_BENCH(nvbench_dense_nn_gemv).set_name("dense_nn_gemv");
NVBENCH_BENCH(nvbench_sparse_nn_spmm).set_name("sparse_nn_spmm").add_int64_axis("sparsity", {50, 55, 60, 65, 70, 75, 80, 85, 90, 95});
NVBENCH_BENCH(nvbench_sparse_nn_spmv).set_name("sparse_nn_spmv").add_int64_axis("sparsity", {50, 55, 60, 65, 70, 75, 80, 85, 90, 95});
