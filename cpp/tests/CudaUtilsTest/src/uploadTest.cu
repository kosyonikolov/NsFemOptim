#include <iostream>
#include <random>
#include <vector>

#include <cu/vec.h>
#include <cu/csr.h>

__global__ void mul10(int * vec, const int size)
{
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = i0; i < size; i += stride)
    {
        vec[i] *= vec[i];
    }
}

void test()
{
    const int n = 100;
    std::vector<int> cpuVec(n);
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 20);
    for (int i = 0; i < n; i++)
    {
        cpuVec[i] = dist(rng);
        std::cout << cpuVec[i] << " ";
    }
    std::cout << "\n";

    cu::vec<int> gpuVec(n);
    gpuVec.uploadAsync(cpuVec);
    // auto rc = cudaMemcpy(gpuVec.get(), cpuVec.data(), n * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
    // if (rc != cudaSuccess)
    // {
    //     std::cerr << "Failed to memcpy: " << rc << "\n";
    //     return;
    // }

    const dim3 blockSize(1);
    const dim3 gridSize(1);
    mul10<<<gridSize, blockSize>>>(gpuVec.get(), gpuVec.size());

    // rc = cudaMemcpy(cpuVec.data(), gpuVec.get(), n * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    // if (rc != cudaSuccess)
    // {
    //     std::cerr << "Failed to memcpy: " << rc << "\n";
    //     return;
    // }
    gpuVec.downloadAsync(cpuVec);

    auto rc = cudaStreamSynchronize(0);
    if (rc != cudaSuccess)
    {
        std::cerr << "Sync failed: " << rc << "\n";
        return;
    }

    for (int i = 0; i < n; i++)
    {
        std::cout << cpuVec[i] << " ";
    }
    std::cout << "\n";
}