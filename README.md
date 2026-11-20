# Experiment 6 – Memory Optimization in CUDA

## Overview
This experiment demonstrates CUDA global-memory access patterns and a basic GPU matrix multiplication kernel. It was executed on an NVIDIA GeForce RTX 5060 Ti using CUDA Toolkit 13.4.

## Hardware and software environment

| Item | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| Reported global memory | 15.93 GB |
| CUDA Toolkit | 13.4 |
| Compiler | NVIDIA CUDA compiler (`nvcc`) |
| Host compiler environment | Visual Studio 2022 Build Tools, x64 Developer Command Prompt |
| Target GPU architecture | `sm_120` |

## Experiment implementation

### Part A – Global memory access
A 4096 × 4096 data set was used. Each input/output array occupies approximately 64 MB.

- **Coalesced access:** adjacent CUDA threads access adjacent elements, allowing memory requests to be combined efficiently.
- **Uncoalesced access:** threads access elements using a transposed/strided index pattern, producing less sequential global-memory access.

Both kernels perform the same simple operation and their results are checked for correctness.

### Part B – Naive matrix multiplication
Two 256 × 256 matrices are multiplied to produce a 256 × 256 result matrix. A two-dimensional block configuration of 16 × 16 threads is used. The GPU result is checked against a CPU-computed reference.

## Build and execution

Run these commands from the project directory in the **x64 Native Tools / Developer Command Prompt for Visual Studio 2022**.

```bat
nvcc --version
```
Displays the installed CUDA compiler version and confirms that the CUDA Toolkit is available.

```bat
cl
```
Checks that the Microsoft C/C++ compiler is available in the Visual Studio developer environment.

```bat
nvcc -O2 -arch=sm_120 memory_optimization.cu -o memory_optimization.exe
```
Compiles the CUDA source file. `-O2` enables compiler optimizations, `-arch=sm_120` targets the RTX 5060 Ti architecture, and `-o` specifies the executable name.

```bat
memory_optimization.exe
```
Runs the experiment and prints the GPU information, kernel timings, and correctness checks.

```bat
memory_optimization.exe > run1_output.txt
memory_optimization.exe > run2_output.txt
memory_optimization.exe > run3_output.txt
```
Runs the executable three times and redirects each run's console output into a separate text file. The `>` operator creates or replaces the specified output file.

```bat
dir
```
Lists the files and folders in the current project directory.

## Results

The following table summarizes the three recorded runs. Times are GPU kernel execution times in milliseconds.

| Run | Coalesced (ms) | Uncoalesced (ms) | Naive matrix multiplication (ms) |
|---|---:|---:|---:|
| Run 1 | 0.348846 | 0.724754 | 0.029901 |
| Run 2 | 0.370523 | 0.725515 | 0.032563 |
| Run 3 | 0.344558 | 0.729056 | 0.032710 |
| **Average** | **0.354642** | **0.726442** | **0.031725** |

### Performance comparison

- Average coalesced access time: **0.354642 ms**
- Average uncoalesced access time: **0.726442 ms**
- In these measurements, coalesced access was approximately **2.05×** faster than uncoalesced access.
- Coalesced access reduced the measured kernel time by approximately **51.18%** compared with uncoalesced access.
- Average naive matrix multiplication kernel time: **0.031725 ms**.

The comparison describes this experiment's workload and GPU. Actual performance depends on the access pattern, cache behavior, compiler, GPU, and workload size.

## Correctness and validation

All three recorded runs reported:
- Coalesced access verification: **PASSED**
- Uncoalesced access verification: **PASSED**
- Matrix multiplication verification: **PASSED**

The first five matrix output values were consistent across the three runs:
`5575.00 5599.00 5595.00 5549.00 5601.00`

## Graphs

- `graphs/memory_access_comparison.png` compares coalesced and uncoalesced access times across the three runs.
- `graphs/matrix_multiplication_performance.png` shows matrix multiplication kernel time for each run.

## Repository structure

```text
Experiment-6-Memory-Optimization-CUDA/
├── memory_optimization.cu
├── README.md
├── results/
│   ├── run1_output.txt
│   ├── run2_output.txt
│   └── run3_output.txt
│   └── comparison_table.csv
├── graphs/
│   ├── memory_access_comparison.png
│   └── matrix_multiplication_performance.png
└── screenshots/
    ├── cuda_setup_and_initial_execution.png
    └── three_runs_and_directory_listing.png
```

## Notes
- The screenshots folder contains the terminal screenshots supplied for this experiment.
- The executable is not included; compile the source on the target system using the command above.
- The recorded timing values are from three separate executions. The separate initial execution visible in the setup screenshot is not included in the three-run averages.
