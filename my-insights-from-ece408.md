---
title: "GPU-Based Parallel Programming: My Insights From the ECE408 Course at UIUC"
date: May 18, 2024
read_time: 7 min
---

As a student at the University of Illinois at Urbana-Champaign (UIUC), I've had the priviledge of taking a variety of challenging and fascinating courses. Among them, ECE408: Applied Parallel Programming has been a game-changer in my academic journey. Guided by Professor Kindratenko, I experienced the complexities and challenges of efficiently implementing parallel programs. While there are countless optimizations to explore, success ultimately hinges on the programmer's ability to navigate trade-offs, understand system overhead, and identify the most effective solutions.

### Understanding Parallel Programming
Parallel programming divides complex tasks into smaller, independent sub-tasks that can be executed at the same time using compute resources like threads. It's a critical solution in today's data-driven world, where exponentially growing datasets make traditional serial processing a bottleneck. By leveraging parallel processing, tasks can be completed quicker and more efficiently, making it indispensable in fields such as scientific computing, machine learning, and big data analytics.

### The CUDA Execution Model: GPU Programming Demystified
A major highlight of ECE408 was learning CUDA (Compute Unified Device Architecture), NVIDIA's framework for parallel programming. GPUs, by design, excel at parallelism with their thousands of lightweight cores working simultaneously. Here's a closer look at key components of CUDA's execution model:
- Threads: Smallest unit of execution that can be scheduled by the GPU
- CUDA Cores: Basic processing units that execute a single thread at a time
- Streaming Multiprocessors (SMs): Manage and execute batches of threads, equipped with shared memory for optimized data access.
- Warp Scheduler: Schedules and dispatches warps (groups of 32 threads) to CUDA cores to execute within each SM

The CUDA execution model involves running highly parallel computations on the GPU while leaving serial or modestly parallel tasks to the CPU. A CUDA kernel is executed as a grid of threads, where all threads run the same kernel code under the Single Processor Multiple Data (SPMD) model. Each thread is assigned a unique index, which is used to compute memory addresses and make control decisions, allowing it to handle different parts of the data or computation independently. The diagram below illustrates the CUDA memory model:

![CUDA Memory Model](https://kdm.icm.edu.pl/Tutorials/GPU-intro/GPU_images/CUDA-memory-model.gif)

### Real-World Application: Optimizing CNNs
One of the most rewarding aspects of ECE408 was my final project, where I implemented and parallelized the forward pass of a LeNet-5 Convolutional Neural Network (CNN). The goal was to minimize the computation time as much as possible, achieving at least a 3x speedup over a naive implementation. This project taught me the practical challenges of optimization. Techniques Used:
- Tiled Shared Memory for Matrix Multiplication: GPUs are often bottlenecked by global memory access latency. By loading tile sub-matrices into shared memory, threads could reuse data within the same block, significantly reducing the need for repeated global memory accesses. Optimizing tile sizes was critical. Larger tiles used more shared memory but reduced memory fetches, while smaller tiles avoided shared memory overflows but increased global memory accesses. Finding the balance was key.
- Matrix Unrolling: The input matrix is essentially unrolled into a 2D matrix where each row corresponds to the elements covered by the filter at a specific position. This allows for more efficient memory access patterns and reduces the overhead of global memory accesses.
- Constant Memory for Masks: Filter masks (weights) used in convolution were stored in constant memory, a small but highly efficient memory space optimized for read-only access by all threads. Filter weights were loaded into constant memory at the start of the kernel execution and reused for each thread, minimizing memory traffic.

```c++
// Example of matrix unrolling, tiling, and constant memory for matrix multiplication
__constant__ float mask[500]; // Constant memory for mask values

// Loop over tiles
for (int i = 0; i < N; i += TILE_SIZE) {
    for (int j = 0; j < N; j += TILE_SIZE) {
        for (int k = 0; k < N; k += TILE_SIZE) {
            #pragma unroll // Unroll the loop for better performance
            for (int ii = i; ii < i + TILE_SIZE; ++ii) {
                #pragma unroll
                for (int jj = j; jj < j + TILE_SIZE; ++jj) {
                    // Compute the product and accumulate
                    float sum = 0.0f; 
                    #pragma unroll 
                    for (int kk = k; kk < k + TILE_SIZE; ++kk) {
                        sum += A[ii * N + kk] * B[kk * N + jj] * mask[kk];
                    }
                    C[ii * N + jj] += sum; // Store the result in C
                }
            }
        }
    }
}
```

Optimizations have to be tailored to the problem size and memory hierarchy and each optimization has its own trade-off that I carefully considered before implementation.

### A Perfect Analogy for Parallel Programming
In conclusion, ECE408 highlighted that parallel programming isn't just about splitting up tasks—it's about understanding bottlenecks, overheads, and trade-offs. Think of a parallel program as a restaurant kitchen. While multiple chefs (threads) can cook different dishes simultaneously, having too many chefs in a small kitchen can lead to chaos (overhead). The key to optimizing parallel programs lies in designing an efficient workflow, ensuring that chefs don't have to wait for ingredients (memory) and work harmonisouly without bumping into each other (synchronization).
