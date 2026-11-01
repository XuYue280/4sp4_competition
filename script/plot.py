import json
import matplotlib.pyplot as plt
import sys
import os

def plot_google_benchmark_median(json_path):
    """
    Parse Google Benchmark JSON output and plot median times.

    Args:
        json_path: Path to the benchmark JSON file
    """
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file {json_path}")
        return

    # TODO: Extract benchmark data and create plots
    # Expected structure: data["benchmarks"] contains list of benchmark results
    # Each result has "name", "median_time", "iterations", etc.

    if "benchmarks" not in data:
        print("Warning: No benchmarks found in JSON file")
        return

    benchmarks = data["benchmarks"]
    names = [b.get("name", "Unknown") for b in benchmarks]
    times = [b.get("median_time", 0) for b in benchmarks]

    plt.figure(figsize=(12, 6))
    plt.bar(names, times)
    plt.ylabel("Time (ns)")
    plt.xlabel("Benchmark Name")
    plt.title("Google Benchmark Median Times")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # plots folder does not exist, create it
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
    plt.savefig("./plots/plot1.png")
    print(f"Plot saved to ./plots/plot1.png")

# Example usage: the input should stay the same. The output plot should be saved in the plots folder.
if __name__ == "__main__":
    json_file = "./logs/project.json"
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    plot_google_benchmark_median(json_file)
