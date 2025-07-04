using Plots
using DelimitedFiles

function plotMesh(nodes::AbstractMatrix{<:Number}, elements::AbstractMatrix{<:Integer})
    #p = scatter(nodes[:,1], nodes[:,2], label="", aspect_ratio=1, markersize=1, markerstrokewidth=0)
    p = Plots.plot(aspect_ratio=1, markersize=1, markerstrokewidth=0)
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

function groupBorderNodesByEntity(border::AbstractMatrix{<:Integer})
    @assert(size(border)[2] == 4)
    maxE = maximum(border[:,1])
    groups = [Vector{Int64}() for i = 1:maxE]
    n = size(border)[1]
    for i = 1:n
        e = border[i,1]
        push!(groups[e], border[i,3])
        push!(groups[e], border[i,4])
    end

    return groups
end

function plotBorderNodes!(p, border::AbstractMatrix{<:Integer}, nodes::AbstractMatrix{<:Number})
    groups = groupBorderNodesByEntity(border)
    n = lastindex(groups)
    for i = 1:n
        g = groups[i]
        if isempty(g)
            continue
        end

        τ = nodes[g, :]
        scatter!(p, τ[:,1], τ[:,2], label = "border $i")
    end
    return p
end

function plotBorderNormals(nodes::AbstractMatrix{<:Number}, border::AbstractMatrix{<:Integer})
    p = Plots.plot()
    n = size(border)[1]
    for i = 1:n
        pts = nodes[border[i,3:4], :]
        plot!(p, pts[:,1], pts[:,2], label="", color=:black)

        mv = pts[2,:] - pts[1,:]
        normal = zeros(2,2)
        normal[1,:] = 0.5 * (pts[1,:] + pts[2,:])
        normal[2,1] = normal[1,1] - mv[2]
        normal[2,2] = normal[1,2] + mv[1]
        plot!(p, normal[:,1], normal[:,2], label="", color=:red)
    end
    return p
end