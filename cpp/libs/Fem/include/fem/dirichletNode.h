#ifndef LIBS_FEM_INCLUDE_FEM_DIRICHLETNODE
#define LIBS_FEM_INCLUDE_FEM_DIRICHLETNODE

namespace fem
{
    struct DirichletNode
    {
        int id;
        float value;
        bool operator<(const DirichletNode & other) const
        {
            return id < other.id;
        }
    };
} // namespace fem

#endif /* LIBS_FEM_INCLUDE_FEM_DIRICHLETNODE */
