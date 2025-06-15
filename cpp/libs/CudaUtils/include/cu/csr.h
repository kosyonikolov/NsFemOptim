#ifndef LIBS_CUDAUTILS_INCLUDE_CU_CSR
#define LIBS_CUDAUTILS_INCLUDE_CU_CSR

#include <linalg/csrMatrix.h>

#include <cu/vec.h>

namespace cu
{
    template <typename F>
    struct csr
    {
        int rows, cols;
        cu::vec<F> values;
        cu::vec<int> column;
        // size = rows + 1, last index is size(coeffs)
        cu::vec<int> rowStart;

        void upload(const linalg::CsrMatrix<F> & m)
        {
            rows = m.rows;
            cols = m.cols;

            values.overwriteUpload(m.values);
            column.overwriteUpload(m.column);
            rowStart.overwriteUpload(m.rowStart);
        }
    };
}

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_CSR */
