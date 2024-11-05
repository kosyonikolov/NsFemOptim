using LinearAlgebra

include("coordTransform.jl")

function calcLocalMassMatrix(nodes::AbstractMatrix{<:Number}, val::Function, xyw::AbstractMatrix{<:Number})
    j = calcRefTriangleTransformJacobian(nodes[:,1], nodes[:,2])
    detJ = det(j)
    first = val(xyw[1,1], xyw[1,2]) .* xyw[1,3]
    result = first * first'
    m = size(xyw)[2]
    for i = 2:m
        curr = val(xyw[i,1], xyw[i,2]) .* xyw[i,3]
        result .+= curr * curr'
    end

    result .*= detJ
    return result
end

function calcLocalLoadVector(nodes::AbstractMatrix{<:Number}, val::Function, f::Function, xyw::AbstractMatrix{<:Number})
    m, b = calcRefTriangleTransform(nodes[:,1], nodes[:,2])
    detJ = det(calcRefTriangleTransformJacobian(nodes[:,1], nodes[:,2]))

    result = val(xyw[1,1:2]) .* (xyw[1,3] * f(m * xyw[1,1:2] .+ b))
    m = size(xyw)[2]
    for i = 2:m
        result .+= val(xyw[i,1:2]) .* (xyw[i,3] * f(m * xyw[i,1:2] .+ b))
    end
    result .*= detJ
    return result 
end