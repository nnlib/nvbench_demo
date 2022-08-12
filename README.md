# Quick Start

```
git clone --recursive <this_project> nvbench_demo

mkdir -p nvbench_demo/build

cd nvbench_demo/build

# Use your actual target architecture(s) or omit the option
# You can look it up at https://developer.nvidia.com/cuda-gpus
#
# Compiling on GeForce GTX 3080 Ti:
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..

make

./example_bench
```
