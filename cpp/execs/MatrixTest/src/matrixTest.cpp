#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename F>
struct CsrMatrix // Implicitly square
{
    int rows, cols;

    // Same size (num maybe-nonzero elems)
    std::vector<F> values;
    std::vector<int> colIdx;
    // size = rows + 1, last index is size(coeffs)
    std::vector<int> rowStart;
};

template <typename F>
std::vector<F> readVector(const std::string & fileName)
{
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error(std::format("Failed to open vector file [{}]", fileName));
    }

    int header[2];
    if (!file.read(reinterpret_cast<char *>(header), sizeof(header)))
    {
        throw std::runtime_error(std::format("Failed to read header from file [{}]", fileName));
    }
    const int elSize = header[0];
    const int size = header[1];
    std::cout << std::format("Element size = {}, size = {}\n", elSize, size);
    if (elSize != sizeof(F))
    {
        throw std::runtime_error(std::format("Failed to vector from file [{}] - element type mismatch", fileName));
    }
    if (size < 0)
    {
        throw std::runtime_error(std::format("Failed to vector from file [{}] - bad size", fileName));
    }

    std::vector<F> result(size);
    if (!file.read(reinterpret_cast<char *>(result.data()), size * elSize))
    {
        throw std::runtime_error(
            std::format("Failed to vector from file [{}] - couldn't read vector values", fileName));
    }

    return result;
}

template <typename F>
CsrMatrix<F> readMatrix(const std::string & fileName)
{
    std::ifstream file(fileName, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error(std::format("Failed to open matrix file [{}]", fileName));
    }

    int header[4];
    if (!file.read(reinterpret_cast<char *>(header), sizeof(header)))
    {
        throw std::runtime_error(std::format("Failed to read header from file [{}]", fileName));
    }
    const int elSize = header[0];
    const int nRows = header[1];
    const int nCols = header[2];
    const int nnz = header[3];
    std::cout << std::format("Element size = {}, rows = {}, cols = {}, nnz = {}\n", elSize, nRows, nCols, nnz);
    if (elSize != sizeof(F))
    {
        throw std::runtime_error(std::format("Failed to read matrix from file [{}] - element type mismatch", fileName));
    }
    if (nRows < 0 || nCols < 0 || nnz < 0)
    {
        throw std::runtime_error(std::format("Failed to read matrix from file [{}] - bad dimensions", fileName));
    }

    CsrMatrix<F> result;
    result.rows = nRows;
    result.cols = nCols;
    result.rowStart.resize(nRows + 1);
    result.colIdx.resize(nnz);
    result.values.resize(nnz);

    if (!file.read(reinterpret_cast<char *>(result.rowStart.data()),
                   result.rowStart.size() * sizeof(result.rowStart[0])))
    {
        throw std::runtime_error(
            std::format("Failed to read matrix from file [{}] - couldn't read rowStart", fileName));
    }
    if (!file.read(reinterpret_cast<char *>(result.colIdx.data()), result.colIdx.size() * sizeof(result.colIdx[0])))
    {
        throw std::runtime_error(std::format("Failed to read matrix from file [{}] - couldn't read colIdx", fileName));
    }
    if (!file.read(reinterpret_cast<char *>(result.values.data()), result.values.size() * sizeof(result.values[0])))
    {
        throw std::runtime_error(std::format("Failed to read matrix from file [{}] - couldn't read values", fileName));
    }

    return result;
}

template <typename F>
void spmv(const CsrMatrix<F> & m, const float * x, float * out)
{
    const int nRows = m.rows;
    for (int i = 0; i < nRows; i++)
    {
        const int j1 = m.rowStart[i + 1];
        F sum = 0;
        for (int j = m.rowStart[i]; j < j1; j++)
        {
            const int col = m.colIdx[j];
            sum += m.values[j] * x[col];
        }
        out[i] = sum;
    }
}

// ||Mx - b||
template <typename F>
double residual(const CsrMatrix<F> & m, const float * x, const float * b)
{
    double errSum = 0;
    const int nRows = m.rows;
    for (int i = 0; i < nRows; i++)
    {
        const int j1 = m.rowStart[i + 1];
        F sum = 0;
        for (int j = m.rowStart[i]; j < j1; j++)
        {
            const int col = m.colIdx[j];
            sum += m.values[j] * x[col];
        }
        const auto delta = sum - b[i];
        errSum += delta * delta;
    }
    return std::sqrt(errSum / nRows);
}

template <typename F>
void gaussSeidelStep(const CsrMatrix<F> & m, F * x, const F * b)
{
    const int nRows = m.rows;
    for (int i = 0; i < nRows; i++)
    {
        const int j1 = m.rowStart[i + 1];
        F diag = 0; // element at (i, i)
        double negSum = 0;
        for (int j = m.rowStart[i]; j < j1; j++)
        {
            const int col = m.colIdx[j];
            if (col == i)
            {
                diag = m.values[j];
            }
            else
            {
                negSum += m.values[j] * x[col];
            }
        }
        x[i] = (b[i] - negSum) / diag;
    }
}

template <typename F>
double gaussSeidel(const CsrMatrix<F> & m, F * x, const F * b, const int maxIters, const double eps)
{
    double lastRes = 0;
    for (int i = 0; i < maxIters; i++)
    {
        gaussSeidelStep(m, x, b);
        lastRes = residual(m, x, b);
        std::cout << std::format("{}: {}\n", i, lastRes);

        if (lastRes < eps)
        {
            break;
        }
    }
    return lastRes;
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./MatrixTest <matrix> <b> <x0> <x>";
    if (argc != 5)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string matrixFname = argv[1];
    const std::string bFname = argv[2];
    const std::string x0Fname = argv[3];
    const std::string xFname = argv[4];

    auto m = readMatrix<float>(matrixFname);
    auto b = readVector<float>(bFname);
    auto x0 = readVector<float>(x0Fname);
    auto x = readVector<float>(xFname);

    const int nRows = m.rows;

    std::vector<float> test(nRows);
    spmv(m, x0.data(), test.data());

    const double initial = residual(m, x0.data(), b.data());
    std::cout << std::format("Initial error: {}\n", initial);

    auto x1 = x0;
    const double gsRes = gaussSeidel(m, x1.data(), b.data(), 100, 1e-4);
    std::cout << std::format("Gauss-Seidel err: {}\n", gsRes);

    return 0;
}
