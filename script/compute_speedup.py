#!/usr/bin/env python3
"""
Compute geometric mean (gmean) speedup per kernel and across all kernels.
Speedup is calculated as: baseline_time / optimized_time
gmean is calculated as: (product of all speedups) ^ (1/n)
Bonus: sparse optimized kernel vs dense baseline on the same shape (SPMM vs GEMM, SPMV vs GEMV).
Reads Google Benchmark JSON (CPU) or NVBench JSON (GPU); the configurations come from the JSON.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

KERNELS = ["gemm", "gemv", "spmm", "spmv"]
BONUS_PAIRS = {"spmm": "gemm", "spmv": "gemv"}
BONUS_TARGET = {"CPU": 1.2, "GPU": 1.1}
EMPTY_TIME = 2e-5  # seconds; if even the slowest config is faster than this, the kernel is treated as empty
UNIT = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def compute_gmean(values):
    """Compute geometric mean of a list of values."""
    if not values or any(v <= 0 for v in values):
        return None
    product = 1.0
    for v in values:
        product *= v
    return product ** (1.0 / len(values))


def extract_benchmark_data(json_data):
    """Extract times (seconds) as {kernel: {variant: {config: [times]}}}, config = benchmark arguments."""
    benchmarks = defaultdict(lambda: {"baseline": defaultdict(list), "optimized": defaultdict(list)})

    for bench in json_data.get("benchmarks") or []:
        name, *args = bench["name"].lower().split("/")
        if name.startswith("bm_"):
            name = name[3:]
        kernel, _, variant = name.partition("_")
        if kernel not in KERNELS or variant not in ("baseline", "optimized"):
            continue

        if "states" in bench:  # NVBench
            for state in bench["states"]:
                config = tuple(int(a["value"]) for a in state.get("axis_values") or [])
                for summary in state.get("summaries") or []:
                    if summary["tag"] == "nv/cold/time/gpu/mean":
                        value = next(float(d["value"]) for d in summary["data"] if d["name"] == "value")
                        benchmarks[kernel][variant][config].append(value)
        elif bench.get("run_type") != "aggregate" and not (bench.get("error_occurred") or bench.get("skipped")):  # Google Benchmark
            config = tuple(int(a) for a in args if a.isdigit())
            benchmarks[kernel][variant][config].append(bench["real_time"] * UNIT[bench["time_unit"]])

    return benchmarks


def median_times(data):
    """Take the median of the repetitions of every config and check that the kernel was implemented."""
    times = {config: statistics.median(t) for config, t in data.items()}
    if not times:
        return times, "missing"
    if min(times.values()) <= 0:
        return times, "zero time"
    if max(times.values()) < EMPTY_TIME:
        return times, "not implemented"
    return times, "ok"


def compute_speedups(benchmarks):
    """Compute speedup (baseline/optimized) for each kernel on the configs both variants ran."""
    speedups = {}

    for kernel in KERNELS:
        baseline_times, baseline_status = median_times(benchmarks[kernel]["baseline"])
        optimized_times, optimized_status = median_times(benchmarks[kernel]["optimized"])
        configs = sorted(baseline_times.keys() & optimized_times.keys())

        kernel_speedups = []
        if baseline_status == optimized_status == "ok":
            kernel_speedups = [baseline_times[c] / optimized_times[c] for c in configs]

        speedups[kernel] = {
            "status": {"baseline": baseline_status, "optimized": optimized_status},
            "individual_speedups": kernel_speedups,
            "gmean_speedup": compute_gmean(kernel_speedups),
            "num_benchmarks": len(kernel_speedups),
            "num_configs": len(baseline_times.keys() | optimized_times.keys()),
        }

    return speedups


def compute_bonus(benchmarks):
    """Speedup of the sparse optimized kernel over the dense baseline (sparse config without its sparsity)."""
    bonus = {}

    for sparse, dense in BONUS_PAIRS.items():
        dense_times, dense_status = median_times(benchmarks[dense]["baseline"])
        sparse_times, sparse_status = median_times(benchmarks[sparse]["optimized"])

        ratios = []
        if dense_status == sparse_status == "ok":
            ratios = [dense_times[c[:-1]] / t for c, t in sparse_times.items() if c[:-1] in dense_times]

        bonus[f"{sparse.upper()} optimized vs {dense.upper()} baseline"] = {
            "status": f"{dense} baseline {dense_status}, {sparse} optimized {sparse_status}",
            "gmean_speedup": compute_gmean(ratios),
            "num_benchmarks": len(ratios),
            "num_configs": len(sparse_times),
        }

    return bonus


def generate_report(speedups, bonus, device, output_file=None):
    """Generate and print/save a detailed report."""
    report = []
    report.append("=" * 80)
    report.append(f"{device} SPEEDUP ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Per-kernel analysis
    report.append("PER-KERNEL SPEEDUP ANALYSIS:")
    report.append("-" * 80)

    for kernel, data in speedups.items():
        status = data["status"]
        report.append(f"\nKernel: {kernel}")
        report.append(f"  Baseline: {status['baseline']}   Optimized: {status['optimized']}")

        if data["gmean_speedup"] is None:
            report.append("  Not scored")
            continue

        report.append(f"  Number of benchmark configurations: {data['num_benchmarks']} (of {data['num_configs']})")
        report.append(f"  Geometric Mean Speedup: {data['gmean_speedup']:.4f}x")
        report.append(f"  Min Speedup: {min(data['individual_speedups']):.4f}x")
        report.append(f"  Max Speedup: {max(data['individual_speedups']):.4f}x")
        report.append(f"  Avg Speedup: {sum(data['individual_speedups']) / len(data['individual_speedups']):.4f}x")

    # Overall analysis
    report.append("")
    report.append("=" * 80)
    report.append("OVERALL SPEEDUP ANALYSIS (across all kernels):")
    report.append("-" * 80)

    scored = {k: d["gmean_speedup"] for k, d in speedups.items() if d["gmean_speedup"] is not None}
    if scored:
        best, worst = max(scored, key=scored.get), min(scored, key=scored.get)
        report.append(f"Overall Geometric Mean Speedup: {compute_gmean(list(scored.values())):.4f}x")
        report.append(f"Number of kernels: {len(scored)} (of {len(speedups)})")
        report.append(f"Best kernel speedup: {scored[best]:.4f}x ({best})")
        report.append(f"Worst kernel speedup: {scored[worst]:.4f}x ({worst})")
    else:
        report.append("Overall Geometric Mean Speedup: n/a (no kernel could be scored)")

    # Bonus analysis
    target = BONUS_TARGET[device]
    report.append("")
    report.append("=" * 80)
    report.append(f"BONUS: SPARSE vs DENSE BASELINE (target {target}x):")
    report.append("-" * 80)

    for pair, data in bonus.items():
        if data["gmean_speedup"] is None:
            report.append(f"{pair}: not scored ({data['status']})")
        else:
            report.append(f"{pair}: {data['gmean_speedup']:.4f}x "
                          f"({data['num_benchmarks']} of {data['num_configs']} configurations)")

    bonus_gmeans = [d["gmean_speedup"] for d in bonus.values()]
    if None in bonus_gmeans:
        report.append("Bonus Geometric Mean Speedup: n/a (needs both pairs)")
    else:
        bonus_gmean = compute_gmean(bonus_gmeans)
        report.append(f"Bonus Geometric Mean Speedup: {bonus_gmean:.4f}x  {'PASS' if bonus_gmean >= target else 'FAIL'}")

    report.append("=" * 80)

    report_text = "\n".join(report)
    print(report_text)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compute_speedup.py <benchmark_json_file> [output_report_file]")
        print("\nExample: python3 compute_speedup.py logs/project.json logs/speedup_report.txt")
        sys.exit(1)

    json_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(json_file).exists():
        print(f"Error: File '{json_file}' not found.")
        sys.exit(1)

    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        sys.exit(1)

    if not isinstance(json_data, dict):
        print("Error: Not a benchmark JSON file.")
        sys.exit(1)

    # Google Benchmark writes a "context" block, NVBench does not
    device = "CPU" if "context" in json_data else "GPU"

    # Extract and process benchmark data
    benchmarks = extract_benchmark_data(json_data)
    speedups = compute_speedups(benchmarks)
    bonus = compute_bonus(benchmarks)

    # Generate and display report
    generate_report(speedups, bonus, device, output_file)

    if output_file:
        print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()
