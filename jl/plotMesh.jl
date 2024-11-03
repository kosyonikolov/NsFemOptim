using Plots
using DelimitedFiles

function plotMesh(nodes::AbstractMatrix{<:AbstractFloat}, elements::AbstractMatrix{<:Integer})
    p = scatter(nodes[:,1], nodes[:,2], label="", aspect_ratio=1, markersize=1, markerstrokewidth=0)
    n, m = size(elements)
    for i = 1:n
        ids = elements[i,:]
        push!(ids, ids[1])
        xs = nodes[ids, 1]
        ys = nodes[ids, 2]
        plot!(p, xs, ys, label="", color=:black, linewidth=0.5)
    end

    return p
end