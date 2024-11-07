using LinearAlgebra

include("coordTransform.jl")

function calcLocalMassMatrix(nodes::AbstractMatrix{<:Number}, val::Function, xyw::AbstractMatrix{<:Number})
    j = calcRefTriangleTransformJacobian(nodes[:,1], nodes[:,2])
    detJ = det(j)
    first = val(xyw[1,1], xyw[1,2])
    result = (first * first') .* xyw[1,3]
    m = size(xyw)[1]
    for i = 2:m
        curr = val(xyw[i,1], xyw[i,2]) 
        result .+= (curr * curr') .* xyw[i,3]
    end

    result .*= abs(detJ)
    return result
end

function calcLocalStiffnessMatrix(nodes::AbstractMatrix{<:Number}, grad::Function, xyw::AbstractMatrix{<:Number})
    j = calcRefTriangleTransformJacobian(nodes)
    B = calcRefTriangleTransformB(nodes)
    BtB = B' * B

    function sample(refPt, w)
        g = grad(refPt)
        return g * BtB * g' .* w
    end

    result = sample(xyw[1,1:2], xyw[1,3])
    n = size(xyw)[1]
    for i = 2:n
        curr = sample(xyw[i,1:2], xyw[i,3])
        result .+= curr
    end

    result .*= abs(det(j))
    return result
end

function calcLocalLoadVector(nodes::AbstractMatrix{<:Number}, val::Function, f::Function, xyw::AbstractMatrix{<:Number})
    M, b = calcRefTriangleTransform(nodes)
    detJ = det(calcRefTriangleTransformJacobian(nodes))

    function sample(refPt, w)
        shapeVal = val(refPt)
        pt = M * refPt .+ b
        fVal = f(pt)
        r = shapeVal .* fVal
        return r .* w
    end

    result = sample(xyw[1,1:2], xyw[1,3])
    # println("INIT $(xyw[1,1:2]) -> $result")
    n = size(xyw)[1]
    for i = 2:n
        # println("$(xyw[i,1:2])")
        curr = sample(xyw[i,1:2], xyw[i,3])
        # println("\told result = $curr")
        # println("\tcurr = $curr")
        result .+= curr
        # println("\tnew result = $result")
    end
    result .*= abs(detJ)
    return result 
end