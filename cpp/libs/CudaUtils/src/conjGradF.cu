#include <cu/conjGradF.h>

#include <format>
#include <stdexcept>
#include <iostream>

namespace cu
{
    ConjugateGradientF::ConjugateGradientF(csrF & mat)
        : spmv(sparse.handle(), mat), r(mat.rows)
    {
        if (mat.rows != mat.cols)
        {
            throw std::invalid_argument("Matrix is not square");
        }
    }

    float ConjugateGradientF::solve(vec<float> & rhs, vec<float> & x, const int maxIters, const float target)
    {
        const int n = spmv.mat.rows;

        // Init: r = rhs - Mx
        auto xd = x.getCuSparseDescriptor();
        auto rhsd = rhs.getCuSparseDescriptor();

        spmv.compute(xd, spmv.b.getCuSparseDescriptor()); // spmv.b now has Mx
        // r = rhs
        auto cuRc = cudaMemcpy(r.get(), rhs.get(), n * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        if (cuRc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("Memcpy failed: {}", cudaGetErrorName(cuRc)));
        }
        // r = r - 1 * Mx
        float negOne = -1.0f;
        float plusOne = 1.0f;
        auto blasRc = cublasSaxpy(blas.handle, n, &negOne, spmv.b.get(), 1, r.get(), 1);
        if (blasRc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cublasSaxpy failed: {}", cublasGetStatusName(blasRc)));
        }

        auto & p = spmv.x;
        auto & d = spmv.b;
        // p = r
        cuRc = cudaMemcpy(p.get(), r.get(), n * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        if (cuRc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("Memcpy failed: {}", cudaGetErrorName(cuRc)));
        }

        float dotR0 = 0;
        float dotR1 = 0;
        float dotDp = 0;

        blasRc = cublasSdot(blas.handle, n, r.get(), 1, r.get(), 1, &dotR0);
        if (blasRc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cublasSdot failed: {}", cublasGetStatusName(blasRc)));
        }

        double lastMse = std::numeric_limits<double>::infinity();
        for (int iter = 0; iter < maxIters; iter++)
        {
            // m.rMult(p, d);
            spmv.compute();

            blasRc = cublasSdot(blas.handle, n, d.get(), 1, p.get(), 1, &dotDp);
            if (blasRc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error(std::format("cublasSdot failed: {}", cublasGetStatusName(blasRc)));
            }

            const float alpha = dotR0 / dotDp;
            // const double dotDp = linalg::dot(d, p);
            // const double alpha = dotR0 / dotDp;

            // Update x and r
            const float negAlpha = -alpha;
            blasRc = cublasSaxpy(blas.handle, n, &alpha, p.get(), 1, x.get(), 1);
            if (blasRc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error(std::format("cublasSaxpy failed: {}", cublasGetStatusName(blasRc)));
            }
            blasRc = cublasSaxpy(blas.handle, n, &negAlpha, d.get(), 1, r.get(), 1);
            if (blasRc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error(std::format("cublasSaxpy failed: {}", cublasGetStatusName(blasRc)));
            }
            // for (int i = 0; i < n; i++)
            // {
            //     x[i] += alpha * p[i];
            //     r[i] -= alpha * d[i];
            // }

            // const double dotR1 = linalg::dot(r, r);
            blasRc = cublasSdot(blas.handle, n, r.get(), 1, r.get(), 1, &dotR1);
            if (blasRc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error(std::format("cublasSdot failed: {}", cublasGetStatusName(blasRc)));
            }
            const double currMse = std::sqrt(dotR1 / n);
            std::cout << "[CGF] " << iter << ": " << currMse << "\n";
            lastMse = currMse;

            // Check for convergence
            if (currMse <= target)
            {
                break;
            }

            // Update direction
            // p = beta * p
            // p = p + r
            const float beta = dotR1 / dotR0;
            blasRc = cublasSscal(blas.handle, n, &beta, p.get(), 1);
            if (blasRc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error(std::format("cublasSscal failed: {}", cublasGetStatusName(blasRc)));
            }
            blasRc = cublasSaxpy(blas.handle, n, &plusOne, r.get(), 1, p.get(), 1);
            if (blasRc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error(std::format("cublasSaxpy failed: {}", cublasGetStatusName(blasRc)));
            }
            // for (int i = 0; i < n; i++)
            // {
            //     p[i] = r[i] + beta * p[i];
            // }

            dotR0 = dotR1;
        }

        return lastMse;
    }
} // namespace cu