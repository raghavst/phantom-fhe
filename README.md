# PhantomFHE: A CUDA-Accelerated Fully Homomorphic Encryption Library

This is a fork of the [PhantomFHE](https://github.com/encryptorion-lab/phantom-fhe)
used for benchmarking.

# Compiling the benchmark

We follow the same steps as mentioned in PhantomFHE docs:

```
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
```

# Run the benchmark

Run the custom benchmark using: 

```
./build/bin/reproduce_bench
```