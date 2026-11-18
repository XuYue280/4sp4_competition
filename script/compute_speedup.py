#!/usr/bin/env python3
"""
Speedup report for the HPP Grand Prix kernels (GEMM, GEMV, SpMM, SpMV) on CPU or GPU.

    speedup = T_baseline / T_optimized    (> 1 means faster than the vendor library)

Per kernel: geometric mean over every configuration in the JSON (the sweep itself is defined in
            main_bench.cpp / main_bench.cu; this script only reads what they wrote).
Overall:    geometric mean of the per-kernel values (each kernel weighs the same).
Bonus:      sparse optimized kernel vs dense vendor library on the same shape (SpMM vs GEMM baseline,
            SpMV vs GEMV baseline); target 1.2x on CPU (MKL) and 1.1x on GPU (cuBLAS).

Reads Google Benchmark JSON (CPU, e.g. logs/project.json) or NVBench JSON (GPU, e.g. logs/project_gpu.json).
Usage: python3 compute_speedup.py <benchmark_json_file> [output_report_file]
"""

import json
import math
import re
import statistics
import sys
from collections import defaultdict

# Parameter order follows the benchmark arguments / NVBench axes:
#   gemm (m, n, k)   gemv (m, n)   spmm (m, n, k, sparsity)   spmv (m, n, sparsity)
NUM_PARAMS = {"gemm": 3, "gemv": 2, "spmm": 4, "spmv": 3}
KERNELS = list(NUM_PARAMS)
# Position of each dimension in the parameter tuple, for the breakdown lines.
DIMS = {
    "gemm": {"size": 0, "bcol": 1},
    "gemv": {"size": 0},
    "spmm": {"size": 0, "bcol": 1, "sparsity": 3},
    "spmv": {"size": 0, "sparsity": 2},
}
# Bytes any real kernel must read at least once: the A matrix (values, plus column indices if sparse).
A_BYTES = {
    "gemm": lambda m, n, k: 4 * m * k,
    "gemv": lambda m, n: 4 * m * n,
    "spmm": lambda m, n, k, s: 8 * m * k * (100 - s) / 100,
    "spmv": lambda m, n, s: 8 * m * n * (100 - s) / 100,
}
# Reading A faster than this is impossible on the cluster (Ada2000 DRAM is ~0.22 TB/s),
# so a variant that does it on its largest config is treated as an empty (unimplemented) kernel.
MAX_BANDWIDTH = 2e12  # bytes/s
# Bonus: sparse kernel -> dense kernel with the same shape (drop the sparsity parameter).
DENSE_OF = {"spmm": "gemm", "spmv": "gemv"}
BONUS_TARGET = {"CPU": 1.2, "GPU": 1.1}

NAME_RE = re.compile(r"(?:BM_)?(gemm|gemv|spmm|spmv)_(baseline|optimized)$", re.IGNORECASE)
UNIT = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def gmean(values):
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def load_gbench(data):
    """Google Benchmark: samples (seconds) per repetition keyed by (kernel, variant, params),
    plus the number of failed runs per (kernel, variant)."""
    samples, failed = defaultdict(list), defaultdict(int)
    for b in data.get("benchmarks") or []:
        if b.get("run_type", "iteration") != "iteration":
            continue  # skip the _mean/_median/_stddev/_cv aggregates
        name, *args = b.get("run_name", b["name"]).split("/")
        m = NAME_RE.match(name)
        if not m:
            continue
        if b.get("error_occurred") or b.get("skipped"):
            failed[m[1].lower(), m[2].lower()] += 1
            continue
        params = tuple(int(a) for a in args if a.isdigit())  # drops "iterations:1", "repeats:5"
        samples[m[1].lower(), m[2].lower(), params].append(b["real_time"] * UNIT[b["time_unit"]])
    return samples, failed


def load_nvbench(data):
    """NVBench: mean GPU time (seconds) of every state keyed by (kernel, variant, params),
    plus the number of failed/skipped states per (kernel, variant)."""
    samples, failed = defaultdict(list), defaultdict(int)
    for b in data.get("benchmarks") or []:
        m = NAME_RE.match(b["name"])
        if not m:
            continue
        for state in b.get("states") or []:
            if state.get("is_skipped") or not state.get("summaries"):
                failed[m[1].lower(), m[2].lower()] += 1  # nvbench records crashes/timeouts as skipped
                continue
            axes = {a["name"]: int(a["value"]) for a in state.get("axis_values") or []}
            known = ("m", "n", "k", "sparsity")  # any extra axis makes the length check in main() reject the state
            params = tuple(axes[k] for k in known if k in axes) + tuple(v for k, v in axes.items() if k not in known)
            for s in state["summaries"]:
                if s["tag"] == "nv/cold/time/gpu/mean":
                    value = next(float(d["value"]) for d in s["data"] if d["name"] == "value")
                    samples[m[1].lower(), m[2].lower(), params].append(value)
    return samples, failed


def check(kernel, times):
    """Return why a variant cannot be scored, or None if it looks implemented."""
    if not times:
        return "missing"
    if min(times.values()) <= 0:
        return "zero time"
    p = max(times, key=lambda p: A_BYTES[kernel](*p))
    bandwidth = A_BYTES[kernel](*p) / times[p]
    if bandwidth > MAX_BANDWIDTH:
        return f"looks empty (reads A at {bandwidth / 1e12:.0f} TB/s on {p})"
    return None


def breakdown(ratios, dims):
    """'by <dim>' lines: gmean of the ratios grouped by each dimension."""
    lines = []
    for dim, i in dims.items():
        groups = defaultdict(list)
        for p, r in ratios.items():
            groups[p[i]].append(r)
        lines.append(f"      by {dim:<9}" + "  ".join(f"{k}:{gmean(v):.2f}x" for k, v in sorted(groups.items())))
    return lines


def kernel_report(kernel, variants, failed):
    base, opt = variants[kernel, "baseline"], variants[kernel, "optimized"]
    why = {"baseline": check(kernel, base), "optimized": check(kernel, opt)}
    label = {v: "ok" if w is None else w.split(" (")[0] for v, w in why.items()}
    paired = sorted(base.keys() & opt.keys())
    configs = f"{len(paired)}/{len(base.keys() | opt.keys())}"
    row = f"{kernel.upper():<6} {label['baseline']:<12} {label['optimized']:<12} {configs:>8}"
    notes = [f"{v}: {w}" for v, w in why.items() if w and w != label[v]]
    notes += [f"{v}: {failed[kernel, v]} runs failed or were skipped (see benchmark log)"
              for v in ("baseline", "optimized") if failed[kernel, v]]
    if len(paired) < len(base.keys() | opt.keys()):
        notes.append(f"{len(base.keys() ^ opt.keys())} configs ran for only one variant (not scored)")
    if any(why.values()) or not paired:
        return None, [f"{row} {'-':>8} {'-':>8} {'-':>8}  NOT SCORED"] + [f"      note: {n}" for n in notes]

    speedups = {p: base[p] / opt[p] for p in paired}
    g = gmean(speedups.values())
    status = "PARTIAL" if notes else "OK"
    lines = [f"{row} {g:>7.3f}x {min(speedups.values()):>7.3f}x {max(speedups.values()):>7.3f}x  {status}"]
    return (g, status), lines + breakdown(speedups, DIMS[kernel]) + [f"      note: {n}" for n in notes]


def bonus_report(variants, target):
    """Sparse optimized kernel vs dense vendor library (baseline) on the same matrix shape."""
    lines, results = [], []
    for sparse, dense in DENSE_OF.items():
        name = f"{sparse.upper()} optimized vs {dense.upper()} baseline"
        d, s = variants[dense, "baseline"], variants[sparse, "optimized"]
        why = [f"{v} {w}" for v, w in ((f"{dense.upper()} baseline", check(dense, d)),
                                       (f"{sparse.upper()} optimized", check(sparse, s))) if w]
        ratios = {p: d[p[:-1]] / s[p] for p in s if p[:-1] in d}
        if why or not ratios:
            lines.append(f"{name:<32} {'-':>8}  NOT SCORED: " + ("; ".join(why) or "no matching shapes"))
            continue
        g = gmean(ratios.values())
        complete = len(ratios) == len(s)
        results.append((g, complete))
        verdict = ("PASS" if g >= target else "FAIL") if complete else "INCOMPLETE"
        lines.append(f"{name:<32} {g:>7.3f}x  {verdict}  ({len(ratios)}/{len(s)} configs have a dense baseline)")
        lines += breakdown(ratios, {"sparsity": DIMS[sparse]["sparsity"]})
    if len(results) == len(DENSE_OF) and all(c for _, c in results):
        g = gmean(r for r, _ in results)
        lines.append(f"BONUS gmean: {g:.3f}x  target {target}x  {'PASS' if g >= target else 'FAIL'}")
    else:
        lines.append(f"BONUS gmean: n/a  target {target}x  (needs both pairs scored on all their configs)")
    return lines


def output(text):
    print(text)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            f.write(text + "\n")
        print(f"Report saved to: {sys.argv[2]}")


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)
    path = sys.argv[1]
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("not a benchmark JSON object")
    except (OSError, ValueError) as e:
        output(f"Error: cannot read '{path}': {e}")  # also replaces a stale report from an earlier run
        sys.exit(1)

    device, fmt, loader = ("CPU", "Google Benchmark", load_gbench) if "context" in data \
        else ("GPU", "NVBench", load_nvbench)
    variants = defaultdict(dict)  # (kernel, variant) -> {params: median seconds}
    bad = 0
    samples, failed = loader(data)
    for (kernel, variant, params), v in samples.items():
        if len(params) == NUM_PARAMS[kernel]:
            variants[kernel, variant][params] = statistics.median(v)
        else:
            bad += 1

    report = [
        "=" * 80,
        f"{device} SPEEDUP REPORT  {path}  ({fmt})",
        "speedup = T_baseline / T_optimized   (> 1 means faster than baseline)",
        "=" * 80,
    ]
    if bad:
        report.append(f"warning: ignored {bad} results whose parameters do not match {NUM_PARAMS}")
    report.append(f"{'kernel':<6} {'baseline':<12} {'optimized':<12} {'configs':>8} {'gmean':>8} {'min':>8} {'max':>8}  status")
    scored = {}
    for kernel in KERNELS:
        result, lines = kernel_report(kernel, variants, failed)
        report += lines
        if result is not None:
            scored[kernel] = result
    report.append("-" * 80)
    if scored:
        groups = {"not scored": [k for k in KERNELS if k not in scored],
                  "partial": [k for k, (_, s) in scored.items() if s == "PARTIAL"]}
        tail = "; ".join(f"{label}: {', '.join(k.upper() for k in ks)}" for label, ks in groups.items() if ks)
        overall = gmean(g for g, _ in scored.values())
        report.append(f"OVERALL gmean over {len(scored)}/{len(KERNELS)} kernels: {overall:.3f}x"
                      + (f"   ({tail})" if tail else ""))
    else:
        report.append("OVERALL gmean: n/a (no kernel could be scored)")
    report += ["=" * 80, "BONUS  sparse optimized vs dense vendor library, same shape"]
    report += bonus_report(variants, BONUS_TARGET[device])
    report.append("=" * 80)
    output("\n".join(report))


if __name__ == "__main__":
    main()
