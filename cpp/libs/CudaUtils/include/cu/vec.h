#ifndef LIBS_CUDAUTILS_INCLUDE_CU_VEC
#define LIBS_CUDAUTILS_INCLUDE_CU_VEC

#include <cassert>
#include <cstddef>
#include <format>
#include <iostream>
#include <stdexcept>

#include <cusparse.h>

#include <utils/concepts.h>

namespace cu
{
    template <typename T>
    class vec
    {
        T * devicePtr = 0;
        size_t length = 0;
        bool externallyOwned = false;

        cusparseDnVecDescr_t cuSparseDescriptor = 0;
        cusparseDnMatDescr_t cuSparseMatDescriptor = 0;

        void resetDescriptor()
        {
            if (cuSparseDescriptor)
            {
                auto rc = cusparseDestroyDnVec(cuSparseDescriptor);
                assert(rc == cusparseStatus_t::CUSPARSE_STATUS_SUCCESS);
                cuSparseDescriptor = 0;
            }
            if (cuSparseMatDescriptor)
            {
                auto rc = cusparseDestroyDnMat(cuSparseMatDescriptor);
                assert(rc == cusparseStatus_t::CUSPARSE_STATUS_SUCCESS);
                cuSparseMatDescriptor = 0;
            }
        }

    public:
        vec() = default;

        // Creates a deep copy
        vec(const vec & other)
        {
            cuSparseDescriptor = 0;
            cuSparseMatDescriptor = 0;
            length = other.length;
            const size_t totalLength = length * sizeof(T);
            auto rc = cudaMalloc(&devicePtr, totalLength);
            if (rc != cudaError_t::cudaSuccess)
            {
                throw std::runtime_error(std::format("[{}:{}] Failed to allocate CUDA memory: {}", __FILE__, __LINE__, static_cast<int>(rc)));
            }

            rc = cudaMemcpy(devicePtr, other.devicePtr, totalLength, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
            if (rc != cudaError_t::cudaSuccess)
            {
                throw std::runtime_error(std::format("[{}:{}] Failed to copy CUDA memory: {}", __FILE__, __LINE__, static_cast<int>(rc)));
            }
        }

        // Wrapper for externally managed memory
        vec(T * extData, const int n)
        {
            devicePtr = extData;
            length = n;
            externallyOwned = true;
        }

        vec(const size_t n)
            : length(n)
        {
            if (n == 0)
            {
                throw std::invalid_argument(std::format("[{}:{}] Size can't be zero", __FILE__, __LINE__));
            }
            const size_t totalLength = n * sizeof(T);
            auto rc = cudaMalloc(&devicePtr, totalLength);
            if (rc != cudaError_t::cudaSuccess)
            {
                throw std::runtime_error(std::format("[{}:{}] Failed to allocate CUDA memory: {}", __FILE__, __LINE__, static_cast<int>(rc)));
            }
        }

        template <u::VectorLike<T> C>
        vec(const C & cpuVec)
        {
            size_t n = cpuVec.size();
            if (n == 0)
            {
                throw std::invalid_argument(std::format("[{}:{}] Size can't be zero", __FILE__, __LINE__));
            }
            const size_t totalLength = n * sizeof(T);
            auto rc = cudaMalloc(&devicePtr, totalLength);
            if (rc != cudaError_t::cudaSuccess)
            {
                throw std::runtime_error(std::format("[{}:{}] Failed to allocate CUDA memory: {}", __FILE__, __LINE__, static_cast<int>(rc)));
            }
            length = n;
            upload(cpuVec.data());
        }

        vec(vec && other)
        {
            if (devicePtr)
            {
                cudaFree(devicePtr);
            }
            devicePtr = other.devicePtr;
            length = other.length;
            other.devicePtr = 0;
            other.length = 0;
        }

        vec & operator=(vec && other) noexcept
        {
            if (this != &other)
            {
                if (devicePtr)
                {
                    cudaFree(devicePtr);
                }
                devicePtr = other.devicePtr;
                length = other.length;
                externallyOwned = other.externallyOwned;
                cuSparseDescriptor = other.cuSparseDescriptor;
                cuSparseMatDescriptor = other.cuSparseMatDescriptor;

                other.devicePtr = 0;
                other.length = 0;
                other.externallyOwned = false;
                other.cuSparseDescriptor = 0;
                other.cuSparseMatDescriptor = 0;
            }
            return *this;
        }

        ~vec()
        {
            resetDescriptor();

            if (externallyOwned)
            {
                externallyOwned = false;
                devicePtr = 0;
                length = 0;
            }
            else if (devicePtr)
            {
                assert(length > 0);
                auto rc = cudaFree(devicePtr);
                if (rc != cudaSuccess)
                {
                    std::cerr << std::format("[{}:{}] Failed to free CUDA memory: {}\n", __FILE__, __LINE__, static_cast<int>(rc));
                }
                else
                {
                    devicePtr = 0;
                    length = 0;
                }
            }
        }

        void memsetZero()
        {
            if (!devicePtr)
            {
                throw std::runtime_error("memsetZero called on an empty vector");
            }
            auto rc = cudaMemset(devicePtr, 0, length * sizeof(T));
            if (rc != cudaError_t::cudaSuccess)
            {
                throw std::runtime_error(std::format("Failed to cuda memset: {}", cudaGetErrorName(rc)));
            }
        }

        void reset()
        {
            resetDescriptor();
            if (externallyOwned)
            {
                externallyOwned = false;
                devicePtr = 0;
                length = 0;
            }
            else if (devicePtr)
            {
                assert(length > 0);
                auto rc = cudaFree(devicePtr);
                if (rc != cudaSuccess)
                {
                    throw std::runtime_error(std::format("[{}:{}] Failed to free CUDA memory: {}\n", __FILE__, __LINE__, static_cast<int>(rc)));
                }
                else
                {
                    devicePtr = 0;
                    length = 0;
                }
            }
        }

        T * release()
        {
            resetDescriptor();
            auto ret = devicePtr;
            devicePtr = 0;
            length = 0;
            externallyOwned = false;
            return ret;
        }

        T * get()
        {
            return devicePtr;
        }

        const T * get() const
        {
            return devicePtr;
        }

        size_t size() const
        {
            return length;
        }

        explicit operator bool() const noexcept
        {
            return devicePtr != nullptr;
        }

        void upload(const T * src)
        {
            if (!devicePtr || !length)
            {
                throw std::invalid_argument(std::format("[{}:{}] Bad upload - devicePtr is null\n", __FILE__, __LINE__));
            }
            auto rc = cudaMemcpy(devicePtr, src, length * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice);
            if (rc != cudaSuccess)
            {
                throw std::invalid_argument(std::format("[{}:{}] Failed to upload to device: {}\n", __FILE__, __LINE__,
                                                        static_cast<int>(rc)));
            }
        }

        void download(T * dst)
        {
            if (!devicePtr || !length)
            {
                throw std::invalid_argument(std::format("[{}:{}] Bad upload - devicePtr is null\n", __FILE__, __LINE__));
            }
            auto rc = cudaMemcpy(dst, devicePtr, length * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
            if (rc != cudaSuccess)
            {
                throw std::invalid_argument(std::format("[{}:{}] Failed to download from device: {}\n", __FILE__, __LINE__,
                                                        static_cast<int>(rc)));
            }
        }

        template <u::VectorLike<T> C>
        void upload(const C & src)
        {
            if (src.size() != length)
            {
                throw std::invalid_argument(std::format("[{}:{}] Size mismatch - got {}, expected {}",
                                                        __FILE__, __LINE__, src.size(), length));
            }
            upload(src.data());
        }

        template <u::VectorLike<T> C>
        void download(C & dst)
        {
            if (dst.size() != length)
            {
                throw std::invalid_argument(std::format("[{}:{}] Size mismatch - got {}, expected {}",
                                                        __FILE__, __LINE__, dst.size(), length));
            }
            download(dst.data());
        }

        void uploadAsync(const T * src, cudaStream_t stream = 0)
        {
            if (!devicePtr || !length)
            {
                throw std::invalid_argument(std::format("[{}:{}] Bad upload - devicePtr is null\n", __FILE__, __LINE__));
            }
            auto rc = cudaMemcpyAsync(devicePtr, src, length * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice, stream);
            if (rc != cudaSuccess)
            {
                throw std::invalid_argument(std::format("[{}:{}] Failed to upload to device: {}\n", __FILE__, __LINE__,
                                                        static_cast<int>(rc)));
            }
        }

        void downloadAsync(T * dst, cudaStream_t stream = 0)
        {
            if (!devicePtr || !length)
            {
                throw std::invalid_argument(std::format("[{}:{}] Bad upload - devicePtr is null\n", __FILE__, __LINE__));
            }
            auto rc = cudaMemcpyAsync(dst, devicePtr, length * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
            if (rc != cudaSuccess)
            {
                throw std::invalid_argument(std::format("[{}:{}] Failed to upload to device: {}\n", __FILE__, __LINE__,
                                                        static_cast<int>(rc)));
            }
        }

        template <u::VectorLike<T> C>
        void uploadAsync(const C & src, cudaStream_t stream = 0)
        {
            if (src.size() != length)
            {
                throw std::invalid_argument(std::format("[{}:{}] Size mismatch - got {}, expected {}",
                                                        __FILE__, __LINE__, src.size(), length));
            }
            uploadAsync(src.data(), stream);
        }

        template <u::VectorLike<T> C>
        void downloadAsync(C & dst, cudaStream_t stream = 0)
        {
            if (dst.size() != length)
            {
                throw std::invalid_argument(std::format("[{}:{}] Size mismatch - got {}, expected {}",
                                                        __FILE__, __LINE__, dst.size(), length));
            }
            downloadAsync(dst.data(), stream);
        }

        template <u::VectorLike<T> C>
        void overwriteUpload(const C & src)
        {
            const size_t n = src.size();
            if (!n)
            {
                throw std::invalid_argument(std::format("[{}:{}] Size can't be 0\n", __FILE__, __LINE__));
            }

            // Release old memory, if any
            resetDescriptor();
            if (devicePtr && !externallyOwned)
            {
                assert(length > 0);
                auto rc = cudaFree(devicePtr);
                if (rc != cudaSuccess)
                {
                    throw std::runtime_error(std::format("[{}:{}] Failed to free CUDA memory: {}\n", __FILE__, __LINE__, static_cast<int>(rc)));
                }
            }

            externallyOwned = false;
            const size_t totalLength = n * sizeof(T);
            auto rc = cudaMalloc(&devicePtr, totalLength);
            if (rc != cudaError_t::cudaSuccess)
            {
                throw std::runtime_error(std::format("[{}:{}] Failed to allocate CUDA memory: {}", __FILE__, __LINE__, static_cast<int>(rc)));
            }
            length = n;
            upload(src.data());
        }

        // Creates if it doesn't exist
        cusparseDnVecDescr_t getCuSparseDescriptor();

        // Get descriptor for a column major matrix
        // rows = n / numCh, cols = numCh
        // Creates the descriptor if it doesn't exist
        cusparseDnMatDescr_t getCuSparseMatDescriptor(const int numCh);
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_VEC */
