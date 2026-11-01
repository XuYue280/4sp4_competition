// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.



#include <nvbench/nvbench.cuh>
#include "kernels.cuh"
#include "gpu_utils.h"


void report_summary(nvbench::state& state)
{
    state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
    state.get_summary("nv/cold/time/cpu/mean").set_string("hide", "");
    state.get_summary("nv/cold/time/cpu/stdev/relative").set_string("hide", "");
}

void nvbench_gemm(nvbench::state& state)
{
    const int n = static_cast<int>(state.get_int64("n"));

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // TODO: IMPLEMENT THIS
        timer.stop();
    });

    // TODO: IMPLEMENT THIS: compare the device result to a reference with a small epsilon

    report_summary(state);
}

void nvbench_gemv(nvbench::state& state)
{
    const int n = static_cast<int>(state.get_int64("n"));

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // TODO: IMPLEMENT THIS
        timer.stop();
    });

    // TODO: IMPLEMENT THIS: compare the device result to a reference with a small epsilon

    report_summary(state);
}

void nvbench_spmm(nvbench::state& state)
{
    const int n = static_cast<int>(state.get_int64("n"));

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // TODO: IMPLEMENT THIS
        timer.stop();
    });

    // TODO: IMPLEMENT THIS: compare the device result to a reference with a small epsilon

    report_summary(state);
}

void nvbench_spmv(nvbench::state& state)
{
    const int n = static_cast<int>(state.get_int64("n"));

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // TODO: IMPLEMENT THIS
        timer.stop();
    });

    // TODO: IMPLEMENT THIS: compare the device result to a reference with a small epsilon

    report_summary(state);
}

NVBENCH_BENCH(nvbench_gemm).set_name("gemm").add_int64_axis("n", {256, 512, 1024, 2048});
NVBENCH_BENCH(nvbench_gemv).set_name("gemv").add_int64_axis("n", {256, 512, 1024, 2048});
NVBENCH_BENCH(nvbench_spmm).set_name("spmm").add_int64_axis("n", {256, 512, 1024, 2048});
NVBENCH_BENCH(nvbench_spmv).set_name("spmv").add_int64_axis("n", {256, 512, 1024, 2048});
