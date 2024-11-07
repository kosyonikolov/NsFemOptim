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

function solveHeatEquation2BordersWithConvection(nodes::AbstractMatrix{<:Number}, elements::AbstractMatrix{<:Integer}, 
                                                 g1::AbstractArray{<:Integer}, g2::AbstractArray{<:Integer},
                                                 flow::Function)
    xyw = getStdTriangleCowperXyw(2)
    m1 = assembleGlobalStiffnessMatrix(nodes, elements, p1Grad, xyw)
    mC = assembleGlobalConvectionMatrix(nodes, elements, p1Value, p1Grad, flow, xyw)
    m1 .+= mC
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

function solveHeatEquationMixedBorders(nodes::AbstractMatrix{<:Number}, elements::AbstractMatrix{<:Integer}, 
                                       flowBorderElements::AbstractMatrix{<:Integer}, 
                                       dirichletBorderElements::AbstractMatrix{<:Integer})
    xyw = getStdTriangleCowperXyw(2)
    xw = getGaussQuadrature(2)
    m1 = assembleGlobalStiffnessMatrix(nodes, elements, p1Grad, xyw)
    b1 = assembleGlobalBorderLoadVector(nodes, flowBorderElements, p1Value1d, p -> 1, xw)

    n = size(nodes)[1]
    borderIds = unique(sort(vec(dirichletBorderElements)))
    intIds = setdiff(1:n, borderIds)

    m = m1[intIds, intIds]
    b = b1[intIds]
    qInt = m \ b

    result = zeros(n)
    result[intIds] .= qInt
    return result
end