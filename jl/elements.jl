# =========== Standard P1 ===========
function p1Value(x::Number, y::Number)
    return [1 - x - y, x, y]
end

function p1Value(p::AbstractVector{<:Number, 1})
    return [1 - p[1] - p[2], p[1], p[2]]
end

function p1Grad()
    return [-1 -1; 1 0; 0 1]
end

# =========== Crouzeix–Raviart ===========
function crValue(x::Number, y::Number)
    return [1 - 2y, -1 + 2x + 2y, 1 - 2x]
end

function crGrad()
    return [0 -2; 2 2; -2 0]
end