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

function assembleGlobalStiffnessMatrix(nodes::AbstractMatrix{<:Number}, elements::AbstractMatrix{<:Integer},
                                       shapeFunction::Function, xyw::AbstractMatrix{<:Number})
    n = size(nodes)[1]
    result = spzeros(n, n)
    m = size(elements)[1]
    for i = 1:m
        τ = elements[i, :]
        localMatrix = calcLocalStiffnessMatrix(nodes[τ, :], shapeFunction, xyw)
        result[τ, τ] .+= localMatrix 
    end
    return result
end

function assembleGlobalConvectionMatrix(nodes::AbstractMatrix{<:Number}, elements::AbstractMatrix{<:Integer},
                                        shape::Function, grad::Function, flow::Function, xyw::AbstractMatrix{<:Number})
    n = size(nodes)[1]
    result = spzeros(n, n)
    m = size(elements)[1]
    for i = 1:m
        τ = elements[i, :]
        localMatrix = calcLocalConvectionMatrix(nodes[τ, :], shape, grad, flow, xyw)
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

function assembleGlobalBorderLoadVector(nodes::AbstractMatrix{<:Number}, elements::AbstractMatrix{<:Integer},
                                        shapeFunction::Function, f::Function, xw::AbstractMatrix{<:Number})
    n = size(nodes)[1]
    b = zeros(n)
    m = size(elements)[1]
    for i = 1:m
        τ = elements[i, :]
        localVec = calcLocalBorderLoadVector(nodes[τ, :], shapeFunction, f, xw)
        b[τ] .+= localVec
    end
    return b
end