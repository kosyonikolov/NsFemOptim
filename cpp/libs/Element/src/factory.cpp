#include <element/factory.h>

#include <element/p0.h>
#include <element/p1.h>
#include <element/p2.h>

namespace el
{
    std::unique_ptr<Element> createElement(const Type type)
    {
        if (type == Type::P0)
        {
            return std::make_unique<P0>();
        }
        if (type == Type::P1)
        {
            return std::make_unique<P1>();
        }
        if (type == Type::P2)
        {
            return std::make_unique<P2>();
        }

        throw std::invalid_argument("Invalid element type");
    }
} // namespace el