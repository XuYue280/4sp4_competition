#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="project"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output="project.%j.%N.out"
#SBATCH -t 00:45:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf


echo "----- Building -----"
# Do not change below, it is fixed for everyone
SHAREDDIR=/home/coe4sp4/

# Source Intel MKL environment
source /opt/intel/oneapi/setvars.sh --force
export PATH=/usr/local/cuda/bin:${PATH}

cmake -S . -B $(pwd)/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${SHAREDDIR}/libpfm4/ -DPROFILING_ENABLED=ON -DUSE_MKL=ON -DOPENMP=ON -DGPU_ENABLED=ON
cmake --build $(pwd)/build -- -j8


echo "---- Copying Weights and Dataset -----"

cp -r /home/coe4sp4/data $(pwd)/

echo "---- Running NN, Python----"
# create python virtual environment
python3 -m venv $(pwd)/venv
source $(pwd)/venv/bin/activate
pip install -r $(pwd)/script/requirements.txt
python3 $(pwd)/script/dense_nn.py
python3 $(pwd)/script/sparsify_weight.py


echo "---- Running CPU Benchmarks ----"

mkdir -p $(pwd)/logs
echo "Running GEMM/GEMV/SpMM/SpMV benchmarks..."
$(pwd)/build/project --benchmark_out="$(pwd)/logs/project.json" --benchmark_out_format=json --benchmark_perf_counters="L1-dcache-loads"

echo "Computing speedup metrics for CPU kernels..."
python3 $(pwd)/script/compute_speedup.py $(pwd)/logs/project.json $(pwd)/logs/cpu_speedup_report.txt


echo "---- Running GPU Benchmarks ----"

echo "Running GEMM/GEMV/SpMM/SpMV GPU benchmarks..."
$(pwd)/build/project_gpu --stopping-criterion entropy --json "$(pwd)/logs/project_gpu.json"

echo "---- Running Tests ----"

$(pwd)/build/test/project_gemm_test
$(pwd)/build/test/project_gemv_test
$(pwd)/build/test/project_spmm_test
$(pwd)/build/test/project_spmv_test

echo "Computing speedup metrics for GPU kernels..."
python3 $(pwd)/script/compute_speedup.py $(pwd)/logs/project_gpu.json $(pwd)/logs/gpu_speedup_report.txt






echo "---- All benchmarks completed ----"
