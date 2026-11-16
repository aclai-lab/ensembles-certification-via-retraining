# Project the traces written by experiments/trace_certification.jl with t-SNE.
#
# Execute using: julia --project=experiments/visualization experiments/visualization/embed.jl [constrained feature]
#
# All points (training set, instances added by both repairs, probe points) are embedded
# together, once, so a point keeps its position at every cycle. Two views are written to
# results/visualization/embedding.json:
# - tsne: a 2-D t-SNE of every standardized feature;
# - exact: x is the raw value of the constrained feature, y a 1-D t-SNE of the other
#   (free) features, so the certified region is exactly a vertical band.

using DelimitedFiles
using JSON3
using Random
using Statistics
using TSne

INPUT = joinpath(pwd(), "results", "visualization")
CONSTRAINED = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1

readcsv(name) = readdlm(joinpath(INPUT, name), ','; header=true)[1]

train = readcsv("train.csv")
probes = Float64.(readcsv("probes.csv"))
added = Dict(r => readcsv("added_$(r).csv") for r in ("original", "honour"))

X_train = Float64.(train[:, 2:end])
nfeatures = size(X_train, 2)
free = setdiff(1:nfeatures, CONSTRAINED)

kinds = String[]
labels = String[]
cycles = Int[]
rows = Matrix{Float64}(undef, 0, nfeatures)

function add_points!(kind, X, label, cycle)
    global rows = vcat(rows, X)
    append!(kinds, fill(kind, size(X, 1)))
    append!(labels, label)
    append!(cycles, cycle)
end

add_points!("train", X_train, string.(Int.(train[:, 1])), zeros(Int, size(X_train, 1)))
for r in ("original", "honour")
    a = added[r]
    if size(a, 1) > 0
        add_points!(r, Float64.(a[:, 3:end]), string.(Int.(a[:, 2])), Int.(a[:, 1]))
    end
end
add_points!("probe", probes, fill("", size(probes, 1)), zeros(Int, size(probes, 1)))

mu = mean(X_train; dims=1)
sigma = [s == 0 ? 1.0 : s for s in std(X_train; dims=1)]
Z = (rows .- mu) ./ sigma

println("embedding $(size(Z, 1)) points")
flush(stdout)

# TSne.jl computes the exact t-SNE, quadratic in the number of points, so embed each
# distinct point once:
# - the three probe levels differ only in the constrained feature, by a fraction of its
#   standard deviation, so the 2-D view embeds the centre level and reuses its position;
# - the 1-D view only sees the free features, so identical free profiles (every probe
#   level, and the real profiles copied by the real-profile repair) share one position.
nprobes = count(==("probe"), kinds)
level = nprobes ÷ 3
probe_rows = findall(==("probe"), kinds)
centre = probe_rows[(level + 1):(2 * level)]
keep = vcat(findall(!=("probe"), kinds), centre)
owner = Dict(r => centre[(k - 1) % level + 1] for (k, r) in enumerate(probe_rows))
pos = Dict(r => j for (j, r) in enumerate(keep))

Random.seed!(1605)
embedded = tsne(Z[keep, :], 2, 0, 1000, 30.0)
full = vcat([embedded[pos[get(owner, i, i)], :]' for i in 1:size(Z, 1)]...)

free_rows = [Tuple(round.(Z[i, free]; digits=8)) for i in 1:size(Z, 1)]
distinct = unique(free_rows)
which = Dict(p => j for (j, p) in enumerate(distinct))
println("embedding $(length(keep)) points in 2-D and $(length(distinct)) free profiles in 1-D")
flush(stdout)
Random.seed!(1605)
embedded_line = tsne(vcat([collect(p)' for p in distinct]...), 1, 0, 1000, 30.0)
line = embedded_line[[which[p] for p in free_rows], :]

@assert !any(isnan, full) && !any(isnan, line)

# read as text: readdlm would parse a string of digits as a number
probe_bits = Dict(
    r => [last(split(l, ",")) for l in readlines(joinpath(INPUT, "probes_$(r).csv"))[2:end]]
    for r in ("original", "honour")
)

open(joinpath(INPUT, "embedding.json"), "w") do io
    JSON3.write(
        io,
        (
            constrained=CONSTRAINED,
            kinds=kinds,
            labels=labels,
            cycles=cycles,
            tsne_x=round.(full[:, 1]; digits=3),
            tsne_y=round.(full[:, 2]; digits=3),
            exact_x=round.(rows[:, CONSTRAINED]; digits=4),
            exact_y=round.(line[:, 1]; digits=3),
            probe_bits=probe_bits,
        ),
    )
end

println("written $(joinpath(INPUT, "embedding.json"))")
