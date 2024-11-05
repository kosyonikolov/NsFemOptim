using SparseArrays

include("localMatrix.jl")

function assembleGlobalMassMatrix(nodes::AbstractMatrix{<:Number}, elements::AbstractMatrix{<:Integer},
                                  shapeFunction::Function, xyw::AbstractMatrix{<:Number})
    n = size(nodes)[1]
    result = spzeros(n, n)
    m = size(elements)[1]
    for i = 1:m
        τ = elements[i, :]
        localMatrix = calcLocalMassMatrix(nodes[τ, :], shapeFunction, xyw)
        result[τ, τ] .+= localMatrix 
    end
    return result
end

function assembleGlobalLoadVector(nodes::AbstractMatrix{<:Number}, elements::AbstractMatrix{<:Integer},
                                  shapeFunction::Function, f::Function, xyw::AbstractMatrix{<:Number})
    n = size(nodes)[1]
    b = zeros(n)
    m = size(elements)[1]
    for i = 1:m
        τ = elements[i, :]
        localVec = calcLocalLoadVector(nodes[τ, :], shapeFunction, f, xyw)
        b[τ] .+= localVec
    end
    return b
end