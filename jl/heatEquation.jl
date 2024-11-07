include("assembly.jl")
include("elements.jl")
include("plotMesh.jl")
include("integration.jl")

function solveHeatEquation2Borders(nodes::AbstractMatrix{<:Number}, elements::AbstractMatrix{<:Integer}, 
                                   g1::AbstractArray{<:Integer}, g2::AbstractArray{<:Integer})
    xyw = getStdTriangleCowperXyw(2)
    m1 = assembleGlobalStiffnessMatrix(nodes, elements, p1Grad, xyw)
    n = size(nodes)[1]

    borderValues = zeros(n)
    borderValues[g2] .= 1
    sub = m1 * borderValues

    intIds = setdiff(1:n, union(g1, g2))
    m = m1[intIds, intIds]
    b = -sub[intIds]
    qInt = m \ b

    result = borderValues
    result[intIds] .= qInt

    return result
end