#!/usr/bin/env python3
"""
Compute geometric mean (gmean) speedup per kernel and across all kernels.
Speedup is calculated as: baseline_time / optimized_time
gmean is calculated as: (product of all speedups) ^ (1/n)
"""

import json
import sys
import math
from collections import defaultdict
from pathlib import Path


def compute_gmean(values):
    """Compute geometric mean of a list of values."""
    if not values or any(v <= 0 for v in values):
        return None
    product = 1.0
    for v in values:
        product *= v
    return product ** (1.0 / len(values))


def extract_benchmark_data(json_data):
    """Extract benchmark data from Google Benchmark JSON output."""
    benchmarks = defaultdict(lambda: {"baseline": [], "optimized": []})

    for bench in json_data.get("benchmarks", []):
        name = bench["name"]
        time_ms = bench.get("real_time", 0)

        # Parse kernel name and variant
        if "_baseline" in name:
            kernel = name.replace("_baseline", "")
            benchmarks[kernel]["baseline"].append(time_ms)
        elif "_optimized" in name:
            kernel = name.replace("_optimized", "")
            benchmarks[kernel]["optimized"].append(time_ms)

    return benchmarks


def compute_speedups(benchmarks):
    """Compute speedup (baseline/optimized) for each kernel."""
    speedups = {}

    for kernel, data in benchmarks.items():
        baseline_times = data["baseline"]
        optimized_times = data["optimized"]

        if len(baseline_times) == len(optimized_times) and baseline_times:
            kernel_speedups = []
            for b_time, o_time in zip(baseline_times, optimized_times):
                if o_time > 0:
                    speedup = b_time / o_time
                    kernel_speedups.append(speedup)

            if kernel_speedups:
                gmean = compute_gmean(kernel_speedups)
                speedups[kernel] = {
                    "individual_speedups": kernel_speedups,
                    "gmean_speedup": gmean,
                    "num_benchmarks": len(kernel_speedups)
                }

    return speedups


def generate_report(speedups, output_file=None):
    """Generate and print/save a detailed report."""
    report = []
    report.append("=" * 80)
    report.append("SPEEDUP ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    all_gmeans = []

    # Per-kernel analysis
    report.append("PER-KERNEL SPEEDUP ANALYSIS:")
    report.append("-" * 80)

    for kernel, data in sorted(speedups.items()):
        gmean = data["gmean_speedup"]
        num_benchmarks = data["num_benchmarks"]
        all_gmeans.append(gmean)

        report.append(f"\nKernel: {kernel}")
        report.append(f"  Number of benchmark configurations: {num_benchmarks}")
        report.append(f"  Geometric Mean Speedup: {gmean:.4f}x")
        report.append(f"  Min Speedup: {min(data['individual_speedups']):.4f}x")
        report.append(f"  Max Speedup: {max(data['individual_speedups']):.4f}x")
        report.append(f"  Avg Speedup: {sum(data['individual_speedups']) / len(data['individual_speedups']):.4f}x")

    # Overall analysis
    report.append("")
    report.append("=" * 80)
    report.append("OVERALL SPEEDUP ANALYSIS (across all kernels):")
    report.append("-" * 80)

    if all_gmeans:
        overall_gmean = compute_gmean(all_gmeans)
        report.append(f"Overall Geometric Mean Speedup: {overall_gmean:.4f}x")
        report.append(f"Number of kernels: {len(all_gmeans)}")
        report.append(f"Best kernel speedup: {max(all_gmeans):.4f}x ({list(speedups.keys())[list(all_gmeans).index(max(all_gmeans))]})")
        report.append(f"Worst kernel speedup: {min(all_gmeans):.4f}x ({list(speedups.keys())[list(all_gmeans).index(min(all_gmeans))]})")

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

    # Extract and process benchmark data
    benchmarks = extract_benchmark_data(json_data)
    speedups = compute_speedups(benchmarks)

    if not speedups:
        print("Warning: No speedup data could be extracted from the benchmark file.")
        print("Make sure the benchmark names follow the convention: <kernel>_baseline and <kernel>_optimized")
        sys.exit(1)

    # Generate and display report
    generate_report(speedups, output_file)

    if output_file:
        print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()

