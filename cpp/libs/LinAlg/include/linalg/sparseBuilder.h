#ifndef LIBS_LINALG_INCLUDE_LINALG_SPARSEBUILDER
#define LIBS_LINALG_INCLUDE_LINALG_SPARSEBUILDER

#include <vector>

namespace linalg
{
    template <typename F>
    struct CsrMatrix;

    template <typename F>
    class SparseMatrixBuilder
    {
    public:
        struct ColPair
        {
            int col;
            F value;
        };

    private:
        int rows = 0;
        int cols = 0;

        std::vector<std::vector<ColPair>> rowPairs;

    public:
        SparseMatrixBuilder() = default;

        SparseMatrixBuilder(const int rows, const int cols);

        void resize(const int newRows, const int newCols);

        void add(const int row, const int col, F value);

        void compressRows();

        const std::vector<std::vector<ColPair>> & getRows() const;

        CsrMatrix<F> buildCsr();

        int numRows() const;

        int numCols() const;
    };
} // namespace linalg

#endif /* LIBS_LINALG_INCLUDE_LINALG_SPARSEBUILDER */
