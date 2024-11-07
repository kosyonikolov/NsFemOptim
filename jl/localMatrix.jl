using LinearAlgebra

include("coordTransform.jl")

function calcLocalMassMatrix(triangle::AbstractMatrix{<:Number}, val::Function, xyw::AbstractMatrix{<:Number})
    j = calcRefTriangleTransformJacobian(triangle[:,1], triangle[:,2])
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

function calcLocalStiffnessMatrix(triangle::AbstractMatrix{<:Number}, grad::Function, xyw::AbstractMatrix{<:Number})
    j = calcRefTriangleTransformJacobian(triangle)
    B = calcRefTriangleTransformB(triangle)
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

function calcLocalConvectionMatrix(triangle::AbstractMatrix{<:Number}, shape::Function, grad::Function, flow::Function,
                                   xyw::AbstractMatrix{<:Number})
    M, b = calcRefTriangleTransform(triangle)
    j = calcRefTriangleTransformJacobian(triangle)
    B = calcRefTriangleTransformB(triangle)

    function sample(refPt, w)
        pt = M * refPt .+ b

        shapeVal = shape(refPt)
        gradVal = grad(refPt) * B
        flowVal = flow(pt)

        return gradVal * flowVal * shapeVal' .* w
    end

    result = sample(xyw[1,1:2], xyw[1,3])
    n = size(xyw)[1]
    for i = 2:n
        curr = sample(xyw[i,1:2], xyw[i,3])
        result .+= curr
    end

    sign = det(j) < 0 ? -1 : 1

    return result * sign
end

function calcLocalLoadVector(triangle::AbstractMatrix{<:Number}, val::Function, f::Function, xyw::AbstractMatrix{<:Number})
    M, b = calcRefTriangleTransform(triangle)
    detJ = det(calcRefTriangleTransformJacobian(triangle))

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

function calcLocalBorderLoadVector(triangle::AbstractMatrix{<:Number}, val::Function, f::Function, xw::AbstractMatrix{<:Number})
    mv = triangle[2,:] - triangle[1,:]
    len = sqrt(sum(mv.^2))

    function sample(refX, w)
        shapeVal = val(refX)
        pt = triangle[1,:] .+ (refX .* mv)
        fVal = f(pt)
        return shapeVal .* (fVal * w)
    end

    result = sample(xw[1,1], xw[1,2])
    n = size(xw)[1]
    for i = 2:n
        result .+= sample(xw[i,1], xw[i,2])
    end

    return result * len
end