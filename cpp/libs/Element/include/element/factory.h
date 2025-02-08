#ifndef LIBS_ELEMENT_INCLUDE_ELEMENT_FACTORY
#define LIBS_ELEMENT_INCLUDE_ELEMENT_FACTORY

#include <memory>

#include <element/element.h>

namespace el
{
    std::unique_ptr<Element> createElement(const Type type);
}

#endif /* LIBS_ELEMENT_INCLUDE_ELEMENT_FACTORY */
