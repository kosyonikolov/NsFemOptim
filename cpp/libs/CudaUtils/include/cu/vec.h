#ifndef LIBS_CUDAUTILS_INCLUDE_CU_VEC
#define LIBS_CUDAUTILS_INCLUDE_CU_VEC

#include <cassert>
#include <cstddef>
#include <format>
#include <iostream>
#include <stdexcept>

#include <utils/concepts.h>

namespace cu
{
    template <typename T>
    class vec
    {
        T * devicePtr = 0;
        size_t length = 0;

    public:
        vec() = default;
        vec(const vec & other) = delete;

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

        vec(vec && other)
        {
            devicePtr = other.devicePtr;
            length = other.length;
            other.devicePtr = 0;
            other.length = 0;
        }

        ~vec()
        {
            if (devicePtr)
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

        void reset()
        {
            if (devicePtr)
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
            auto ret = devicePtr;
            devicePtr = 0;
            length = 0;
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
                throw std::invalid_argument(std::format("[{}:{}] Failed to upload to device: {}\n", __FILE__, __LINE__,
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
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_VEC */
