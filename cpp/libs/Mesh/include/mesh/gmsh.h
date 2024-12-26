#ifndef INCLUDE_MESH_GMSH
#define INCLUDE_MESH_GMSH

#include <map>
#include <optional>

#include <mesh/triMesh.h>

namespace mesh
{
    struct SectionedText
    {
        std::map<std::string, int> name2id;
        std::vector<std::string> names;
        std::vector<std::string> content;

        std::optional<std::string> contentOf(const std::string & name)
        {
            const auto it = name2id.find(name);
            if (it == name2id.end())
            {
                return {};
            }
            return content[it->second];
        }
    };

    struct ParsedNode
    {
        int id;
        double x, y, z;
    };

    struct NodeSection
    {
        std::vector<ParsedNode> nodes;
    };

    struct ParsedElement
    {
        int id;
        int entity;
        std::vector<int> points;
    };

    struct ElementSection
    {
        std::vector<ParsedElement> elements;
    };

    struct PhysicalName
    {
        int dimension;
        int tag;
        std::string name;
    };

    struct PhysicsSection
    {
        std::vector<PhysicalName> names;
    };

    struct PointEntity
    {
        int tag;
        double x, y, z;
        std::vector<int> physicalTags;
    };

    struct CurveEntity
    {
        int tag;
        double minX, minY, minZ;
        double maxX, maxY, maxZ;
        std::vector<int> physicalTags;
        std::vector<int> boundingPoints;
    };

    struct SurfaceEntity
    {
        int tag;
        double minX, minY, minZ;
        double maxX, maxY, maxZ;
        std::vector<int> physicalTags;
        std::vector<int> boundingCurves;
    };

    struct VolumeEntity
    {
        int tag;
        double minX, minY, minZ;
        double maxX, maxY, maxZ;
        std::vector<int> physicalTags;
        std::vector<int> boundingSurfaces;
    };

    struct EntitySection
    {
        std::vector<PointEntity> points;
        std::vector<CurveEntity> curves;
        std::vector<SurfaceEntity> surfaces;
        std::vector<VolumeEntity> volumes;
    };

    struct Gmsh
    {
        NodeSection nodeSection;
        ElementSection elementSection;
        PhysicsSection physicsSection;
        EntitySection entitySection;
    };

    void dumpPointsAndElements2d(const std::string & prefix, std::vector<ParsedNode> & nodes,
                                 std::vector<ParsedElement> & elements);

    void dumpPhysicalGroups(const std::string & fileName, const PhysicsSection & phys);

    Gmsh parseGmsh(const std::string & meshFileName);
} // namespace mesh

#endif
