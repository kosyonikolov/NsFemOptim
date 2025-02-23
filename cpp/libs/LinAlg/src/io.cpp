#include <linalg/io.h>

#include <fstream>
#include <stdexcept>

namespace linalg
{
    template <typename F>
    int dtypeId();

    template <>
    int dtypeId<float>()
    {
        return 0;
    }

    template <>
    int dtypeId<double>()
    {
        return 1;
    }

    template <typename T>
    inline void writeVal(std::ostream & stream, const T & t)
    {
        stream.write(reinterpret_cast<const char *>(&t), sizeof(t));
    }

    template <typename T>
    inline void writeVec(std::ostream & stream, const std::vector<T> & v)
    {
        stream.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T));
    }

    template <typename T>
    bool readVal(std::istream & stream, T & outVal)
    {
        return stream.read(reinterpret_cast<char*>(&outVal), sizeof(T)) ? true : false;
    }

    template <typename T>
    bool readVec(std::istream & stream, std::vector<T> & outVec, const int n)
    {
        outVec.resize(n);
        return stream.read(reinterpret_cast<char*>(outVec.data()), n * sizeof(T)) ? true : false;
    }

    template <typename F>
    void write(std::ostream & stream, const CsrMatrix<F> & m)
    {
        // Sanity checks
        if (m.rowStart.size() != m.rows + 1)
        {
            throw std::invalid_argument(std::format("{}: Bad size of rowStart [{}] - expected {}",
                                                    __FUNCTION__, m.rowStart.size(), m.rows + 1));
        }
        if (m.values.size() != m.column.size())
        {
            throw std::invalid_argument(std::format("{}: Bad size of colIdx [{}] - expected {}",
                                                    __FUNCTION__, m.column.size(), m.values.size()));
        }

        const int id = dtypeId<F>();
        writeVal(stream, id);
        writeVal(stream, m.rows);
        writeVal(stream, m.cols);

        const int nnz = m.values.size();
        writeVal(stream, nnz);
        writeVec(stream, m.values);
        writeVec(stream, m.column);
        writeVec(stream, m.rowStart);
    }

    template <typename F>
    void write(std::ostream & stream, const std::vector<F> & v)
    {
        const int id = dtypeId<F>();
        writeVal(stream, id);

        const int n = v.size();
        writeVal(stream, n);
        writeVec(stream, v);
    }

    template <typename F>
    CsrMatrix<F> readCsr(std::istream & stream)
    {
        const int expectedId = dtypeId<F>();
        int id;
        if (!readVal(stream, id) || id != expectedId)
        {
            throw std::runtime_error("Failed to read CSR matrix: Bad dtype");
        }
        
        int rows, cols, nnz;
        if (!readVal(stream, rows) || !readVal(stream, cols) || !readVal(stream, nnz))
        {
            throw std::runtime_error("Failed to read CSR matrix: Couldn't read header");
        }
        if (rows < 1 || cols < 1 || nnz < 1)
        {
            throw std::runtime_error("Failed to read CSR matrix: Bad header values");
        }

        CsrMatrix<F> result;
        result.rows = rows;
        result.cols = cols;
        if (!readVec(stream, result.values, nnz) || !readVec(stream, result.column, nnz) || !readVec(stream, result.rowStart, rows + 1))
        {
            throw std::runtime_error("Failed to read CSR matrix: Couldn't read data vectors");
        }
        if (result.rowStart.back() != nnz)
        {
            throw std::runtime_error("Failed to read CSR matrix: Bad end of rowStart");
        }

        return result;
    }

    template <typename F>
    std::vector<F> readVec(std::istream & stream)
    {
        const int expectedId = dtypeId<F>();
        int id;
        if (!readVal(stream, id) || id != expectedId)
        {
            throw std::runtime_error("Failed to read vector: Bad dtype");
        }
        
        int n;
        if (!readVal(stream, n))
        {
            throw std::runtime_error("Failed to read vector: Couldn't read header");
        }
        if (n < 1)
        {
            throw std::runtime_error("Failed to read vector: Bad size");
        }

        std::vector<F> result;
        if (!readVec(stream, result, n))
        {
            throw std::runtime_error("Failed to read vector: Couldn't read data vector");
        }

        return result;
    }

    template <typename F>
    void write(const std::string & fileName, const CsrMatrix<F> & m)
    {
        std::ofstream file(fileName, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error(std::format("Failed to open output file {}", fileName));
        }
        write(file, m);
    }

    template <typename F>
    void write(const std::string & fileName, const std::vector<F> & v)
    {
        std::ofstream file(fileName, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error(std::format("Failed to open output file {}", fileName));
        }
        write(file, v);
    }

    template <typename F>
    CsrMatrix<F> readCsr(const std::string & fileName)
    {
        std::ifstream file(fileName, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error(std::format("Failed to open input file {}", fileName));
        }
        return readCsr<F>(file);
    }

    template <typename F>
    std::vector<F> readVec(const std::string & fileName)
    {
        std::ifstream file(fileName, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error(std::format("Failed to open input file {}", fileName));
        }

        return readVec<F>(file);
    }

    template void write(const std::string & fileName, const CsrMatrix<float> & m);
    template void write(const std::string & fileName, const CsrMatrix<double> & m);

    template void write(const std::string & fileName, const std::vector<float> & m);
    template void write(const std::string & fileName, const std::vector<double> & m);

    template CsrMatrix<float> readCsr(const std::string & fileName);
    template CsrMatrix<double> readCsr(const std::string & fileName);

    template std::vector<float> readVec(const std::string & fileName);
    template std::vector<double> readVec(const std::string & fileName);
} // namespace linalg