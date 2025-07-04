#ifndef LIBS_LINALG_INCLUDE_LINALG_SPARSEDOKBUILDER
#define LIBS_LINALG_INCLUDE_LINALG_SPARSEDOKBUILDER

#include <span>
#include <vector>

namespace linalg
{
    template <typename F>
    struct CsrMatrix;

    template <typename F>
    struct Triplet
    {
        int row;
        int col;
        F value;
    };

    struct Coordinate
    {
        int row;
        int col;
    };

    template <typename F>
    class SparseMatrixDokBuilder
    {
    private:
        int rows = 0;
        int cols = 0;

        std::vector<Triplet<F>> triplets;

    public:
        SparseMatrixDokBuilder() = default;

        SparseMatrixDokBuilder(const int rows, const int cols);

        void resize(const int newRows, const int newCols);

        void add(const int row, const int col, F value);

        void add(std::span<const SparseMatrixDokBuilder<F> *> others);

        void compress();

        CsrMatrix<F> buildCsr();

        CsrMatrix<F> buildCsr2();

        int numRows() const;

        int numCols() const;
    };

    class SparseMatrixPrototypeBuilder
    {
    private:
        int rows = 0;
        int cols = 0;

        std::vector<Coordinate> coords;

    public:
        SparseMatrixPrototypeBuilder() = default;

        SparseMatrixPrototypeBuilder(const int rows, const int cols);

        void resize(const int newRows, const int newCols);

        void add(const int row, const int col);

        void add(std::span<const SparseMatrixPrototypeBuilder *> others);

        void compress();

        template <typename F>
        CsrMatrix<F> buildCsrPrototype();

        template <typename F>
        CsrMatrix<F> buildCsrPrototype2();

        int numRows() const;

        int numCols() const;
    };
} // namespace linalg

#endif /* LIBS_LINALG_INCLUDE_LINALG_SPARSEDOKBUILDER */
