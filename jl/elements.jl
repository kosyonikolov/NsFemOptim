
# =========== Crouzeix–Raviart ===========
function crShape(x::Number, y::Number)
    return [1 - 2y, -1 + 2x + 2y, 1 - 2x]
end

function crGrad()
    return [0 -2; 2 2; -2 0]
end