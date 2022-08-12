#include <nvbench/nvbench.cuh>

#include <cuda/std/chrono>

#include <cuda_runtime.h>

#include <nvbench/nvbench.cuh>

// Grab some testing kernels from NVBench:
#include <nvbench/test_kernels.cuh>

// Thrust vectors simplify memory management:
#include <thrust/device_vector.h>


__global__ void sleep_kernel(nvbench::int64_t microseconds) {
  const auto start = cuda::std::chrono::high_resolution_clock::now();
  const auto target_duration = cuda::std::chrono::microseconds(microseconds);
  const auto finish = start + target_duration;

  auto now = cuda::std::chrono::high_resolution_clock::now();
  while (now < finish) {
    now = cuda::std::chrono::high_resolution_clock::now();
  }
}

void sleep_benchmark(nvbench::state &state) {
  const auto duration_us = state.get_int64("Duration (us)");
  state.exec([&duration_us](nvbench::launch &launch) {
    sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(duration_us);
  });
}

void throughput_bench(nvbench::state &state)
{
  // Allocate input data:
  const std::size_t num_values = 640 * 1024 * 1024 / sizeof(nvbench::int32_t);
  thrust::device_vector<nvbench::int32_t> input(num_values);
  thrust::device_vector<nvbench::int32_t> output(num_values);

  // Provide throughput information:
  state.add_element_count(num_values, "NumElements");
  state.add_global_memory_reads<nvbench::int32_t>(num_values, "DataSize");
  state.add_global_memory_writes<nvbench::int32_t>(num_values);

  state.exec([&input, &output, num_values](nvbench::launch &launch) {
    nvbench::copy_kernel<<<256, 256, 0, launch.get_stream()>>>(
      thrust::raw_pointer_cast(input.data()),
      thrust::raw_pointer_cast(output.data()),
      num_values);
  });
}

NVBENCH_BENCH(sleep_benchmark)
    .add_int64_axis("Duration (us)", nvbench::range(0, 100, 5))
    .set_timeout(1); // Limit to one second per measurement.

NVBENCH_BENCH(throughput_bench);
