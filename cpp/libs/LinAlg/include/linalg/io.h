#ifndef LIBS_LINALG_INCLUDE_LINALG_IO
#define LIBS_LINALG_INCLUDE_LINALG_IO

#include <istream>
#include <ostream>
#include <string>

#include <linalg/csrMatrix.h>

namespace linalg
{
    template <typename F>
    void write(std::ostream & stream, const CsrMatrix<F> & m);

    template <typename F>
    void write(std::ostream & stream, const std::vector<F> & v);

    template <typename F>
    CsrMatrix<F> readCsr(std::istream & stream);

    template <typename F>
    std::vector<F> readVec(std::istream & stream);

    template <typename F>
    void write(const std::string & fileName, const CsrMatrix<F> & m);

    template <typename F>
    void write(const std::string & fileName, const std::vector<F> & v);

    template <typename F>
    CsrMatrix<F> readCsr(const std::string & fileName);

    template <typename F>
    std::vector<F> readVec(const std::string & fileName);
} // namespace linalg

#endif /* LIBS_LINALG_INCLUDE_LINALG_IO */
