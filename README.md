# 🏎️ HPP Grand Prix: High-Performance Programming Competition

The goal is to go beyond the baseline performance of dense linear algebra libraries (Intel MKL and NVIDIA cuBLAS/cuSPARSE) by implementing **sparse linear algebra kernels**. You will implement 4 kernels:
1. **GEMM:** Dense matrix-matrix multiplication
2. **GEMV:** Dense matrix-vector multiplication
3. **SpMM:** Sparse matrix-matrix multiplication
4. **SpMV:** Sparse matrix-vector multiplication

---

## DEADLINE: 2026-12-06 23:59:59

## 🏆 Competition Leaderboard

This is not just an assignment—it's a **COMPETITION**! Your implementations will be ranked against:
- **CPU Baseline:** Intel MKL (Matrix Kernel Library)
- **GPU Baseline:** NVIDIA cuBLAS and cuSPARSE


### 📊 Metrics Tracked
- **Per-Kernel Speedup:** Geometric mean speedup for each operation (GEMM, GEMV, SpMM, SpMV)
- **Cross-Kernel Speedup:** Overall geometric mean speedup across all 4 kernels
- **Scalability:** Performance across 7 different matrix sizes (256 to 16384)
- **Sparsity Efficiency:** Speedup improvement at 50%, 70%, 80%, and 90% sparsity levels

---

## How Speedup Is Measured

1. **Speedup is computed from the benchmark results.** After `build_run.sh` runs the benchmarks, it calls `script/compute_speedup.py`. The script reads the results (`logs/project.json` for CPU, `logs/project_gpu.json` for GPU) and computes:

   ```
   speedup = baseline geometric mean / optimized geometric mean
   ```

   The report gives:
   - the per-kernel and cross-kernel speedups;
   - the bonus comparison of the sparse kernels against the dense baselines;
   - the report will show `not implemented` for kernels that are not implemented.

   The results are printed in the `.out` file and also saved to `logs/cpu_speedup_report.txt` and `logs/gpu_speedup_report.txt`.

2. **Sweep sizes.** CPU and GPU use the same sweep for matrix dimensions:
   - `m` ∈ {256, 512, 1024, 2048, 4096, 8192}
   - `k` ∈ {32, 64, 256, 512} 
   - sparsity ∈ {50%, 70%, 80%, 90%}

   GEMM and SpMM compute `C = A @ B`, where `A` is `m × k` and `B` is `k × m`. GEMV and SpMV compute `y = A @ x`, where `A` is `m × m` and `x` is `m × 1`. For SpMM and SpMV, `A` is sparse.

   Configurations per kernel (run for both the baseline and your optimized kernel):

   | Kernel | Swept parameters   | Configurations  |
   |--------|--------------------|-----------------|
   | GEMM   | `m`, `k`           | 7 × 4 = 28      |
   | GEMV   | `m`                | 7               |
   | SpMM   | `m`, `k`, sparsity | 7 × 4 × 4 = 112 |
   | SpMV   | `m`, sparsity      | 7 × 4 = 28      |

3. **Recommendations.** While debugging, or if your kernels exceed the 45-minute job limit, you may comment out the larger dimension. For example, comment out the `->Args(...)` lines in `main_bench.cpp` for CPU and the `add_int64_axis(...)` values in `main_bench.cu` for GPU.




## 🛠️ Logging in to the ECE Cluster
The ECE computer server is a local server used for this course. The ECE server
has 6 nodes, each with a 20-core Intel CPU and an NVIDIA Ada2000 GPU. The ECE server
uses a SLURM scheduler. SLURM (Simple Linux Utility for Resource Management)
is an open-source workload manager that allocates exclusive resources (computer nodes)
to users for a specific duration to run their tasks. It manages a queue of jobs,
ensuring that resources are used efficiently and jobs are run fairly. You will
need a username to log in to the server. Your username and password are provided to you.
With that, you will need to SSH to the server using the
following command:

 ```
 ssh <username>@srv-cad.ece.mcmaster.ca
 ```

***

## ➕ Cloning the Repository

This step explains how to clone the repository. You will use **`git`**, a version
control system, to clone the repository and push your code to it. To clone
the repository, you will need to use the following command:


```
git clone https://github.com/4sp4-fall-2026/<repo name>.git
```

Where `repo name` is your repository. The repository is private, so you may get an
error when you enter your username and password. To solve this, you will need to
use a **Personal Access Token (PAT)** instead of your password.


***

## 🚀 Building and running the starter code

This section explains how you can build and run the code on the
ECE cluster. First, go to where the project is cloned:

```
cd <where/the/repo/is/cloned>
```
All necessary instructions to build and run the code are
provided in the `build_run.sh` file.
You can open the file using a text editor like `nano` or `vim`,
e.g., `vim build_run.sh`.  Use the following command to build
and run the code:

```
sbatch build_run.sh
```

This command submits the `build_run.sh` script to the SLURM scheduler.
The instructions in the `build_run.sh` file are unix commands that will be executed on a compute node. The output of
the job will be saved in a `*.out` file.

* **Never use `bash`** to run your code on the login node.
  Use `sbatch` instead. `bash` runs the code directly on
  the login node, while `sbatch` submits the job to the SLURM
  scheduler, which runs it on an available compute node.
* You should **never** run code on the login node unless it
  takes less than 30 seconds.
* You can check the status of your job using `squeue -u <username>`.


***


## ✅ Tasks

- First pick a cool name for your group and update "GroupName.txt" in the root of the repository.
- Do not change anything else except for the 4 functions and 4 kernels that you are required to implement. 


### Evaluation

You can see your score, there will be three gmeans speedups. 
We use the overall gmean speedup to rank the submissions. 


### Submission
Follow these guidelines for a successful submission:
* First, please implement all TODOs in the code and remove all comments that
  start with `TODO`.
* Push all your code to the main branch of your repository before the deadline.
* Ensure your code compiles and runs on the ECE cluster. Submissions that fail to
  compile or run will receive a grade of zero. The TA will only use the `build_run.sh` file for
  building and running your code. You may want to make a copy of the script for your final testing.
* The build_run.sh script's execution should only output results from Google Test and
  Google Benchmark. Do not include any extra output from cout or printf. This can negatively
  impact your grade, and regrade requests for this reason will not be accepted.
* If your code fails in any of the provided test cases, you will not be considered for the competition.


