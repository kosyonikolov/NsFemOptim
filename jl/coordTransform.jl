using LinearAlgebra

# Compute M,b such that M * [u, v] + b = [x, y]
# [u, v] are located in the reference triangle ([0,1]^2)
# [x, y] are in {xs} x {ys}
# [0, 0] -> (x1, y1), [1, 0] -> (x2, y2), [0, 1] -> (x3, y3)
function calcRefTriangleTransform(xs::AbstractVector{<:Number}, ys::AbstractVector{<:Number})
    @assert(lastindex(xs) == 3)
    @assert(lastindex(ys) == 3)

    m = zeros(2,2)
    b = [xs[1], ys[1]]

    m[1,1] = xs[2] - xs[1]
    m[1,2] = xs[3] - xs[1]
    m[2,1] = ys[2] - ys[1]
    m[2,2] = ys[3] - ys[1]

    return m, b
end

function calcRefTriangleTransformJacobian(xs::AbstractVector{<:Number}, ys::AbstractVector{<:Number})
    @assert(lastindex(xs) == 3)
    @assert(lastindex(ys) == 3)

    j = zeros(2,2)

    j[1,1] = xs[2] - xs[1]
    j[2,1] = xs[3] - xs[1]
    j[1,2] = ys[2] - ys[1]
    j[2,2] = ys[3] - ys[1]

    return j
end

# The matrix used for gradient computation
function calcRefTriangleTransformB(xs::AbstractVector{<:Number}, ys::AbstractVector{<:Number})
    @assert(lastindex(xs) == 3)
    @assert(lastindex(ys) == 3)

    b = zeros(2,2)

    b[1,1] = ys[3] - ys[1]
    b[1,2] = ys[1] - ys[2]
    b[2,1] = xs[1] - xs[3]
    b[2,2] = xs[2] - xs[1]

    return b
end