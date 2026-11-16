# Record how certification evolves, cycle by cycle, for the original repair and for the
# real-profile repair (see `honour_distribution` in `experiment`), on the same split.
#
# Execute using: julia --project=. experiments/trace_certification.jl [nhparam] [lcomplexity]
#
# For each repair it writes, under results/visualization:
# - added_<repair>.csv: every instance added to the training set, with the cycle it was
#   added after;
# - probes_<repair>.csv: for every cycle, one 0/1 digit per probe point, 1 when the model
#   trained at that cycle already gives the certified class.
# It also writes train.csv (the original training set) and probes.csv. The probe points
# are every training instance moved into the constrained region, at the lower bound, the
# centre and the upper bound of each constrained feature: they are what certification has
# to cover, seen through a finite sample.
#
# Then run experiments/visualization/embed.jl to project everything with t-SNE.

using FormalVerificationRandomForests
using DecisionTree
using DelimitedFiles
using Random
using SoleData
using SoleLogics
using SoleModels

DATA_NAME = "breastcancer"
MODEL_TYPE = :decision_forest
NTREES = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
LCOMPLEXITY = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1

DATA_PATH = joinpath(pwd(), "dataset")
OUTPUT_PATH = joinpath(pwd(), "results", "visualization")
mkpath(joinpath(OUTPUT_PATH, "smt2dump"))

include(joinpath(pwd(), "experiments", "configurations", "configuration_$(DATA_NAME).jl"))

lconstraint = exp_lconstraints[LCOMPLEXITY]
rconstraint = exp_rconstraints[1]
constrained = constrained_indexes(lconstraint)
target = string(Int(SoleData.threshold(SoleLogics.value(first(atoms(rconstraint))))))

"""
Same load, shuffle and 80/20 split as the first repetition of run_experiment.jl.
"""
function split_dataset()
    Random.seed!(exp_seed)
    X, y = load_dataset(first(exp_datasets); root_path=DATA_PATH)
    perm = randperm(size(X, 1))
    ntrain = floor(Int, 0.8 * size(X, 1))
    return X[perm[1:ntrain], :], y[perm[1:ntrain]]
end

X_train, y_train = split_dataset()

# lower bound, centre and upper bound of each constrained feature
bounds = Dict{Int,Vector{Float64}}()
for a in atoms(lconstraint)
    cond = SoleLogics.value(a)
    i = cond.metacond.feature.i_variable
    push!(get!(bounds, i, Float64[]), SoleData.threshold(cond))
end
levels_per_feature = Dict(i => [minimum(b), (minimum(b) + maximum(b)) / 2, maximum(b)] for (i, b) in bounds)

probes = vcat(
    [
        begin
            p = copy(X_train)
            for i in constrained
                p[:, i] .= levels_per_feature[i][level]
            end
            p
        end for level in 1:3
    ]...,
)

function trace(honour::Bool)
    cycle_sizes = Int[]
    cycle_bits = String[]
    final_X, final_y = nothing, nothing

    function on_cycle(numcycles, model, X, y)
        push!(cycle_sizes, size(X, 1))
        preds = apply_forest(model, probes)
        push!(cycle_bits, join(p == target ? '1' : '0' for p in preds))
        final_X, final_y = X, y
        return nothing
    end

    # restore the random state the driver has right before calling `experiment`
    X, y = split_dataset()
    response = experiment(
        X,
        y,
        first(exp_datasets),
        exp_type;
        smt2filename=joinpath(OUTPUT_PATH, "smt2dump", "trace.smt2"),
        model_type=MODEL_TYPE,
        forest_max_depth=first(exp_maxdepth),
        n_trees=NTREES,
        lconstraint_complexity=LCOMPLEXITY,
        rconstraint_complexity=1,
        default_lconstraints=exp_lconstraints,
        default_rconstraints=exp_rconstraints,
        batch_size=1,
        silent=true,
        honour_distribution=honour,
        on_cycle=on_cycle,
        root_path=DATA_PATH,
    )
    isnothing(response) && error("the experiment did not return a model")

    name = honour ? "honour" : "original"
    ntrain = size(X_train, 1)

    # cycle c is the model trained after c repair cycles; the rows it was trained on
    # beyond the previous cycle's ones are the ones added by cycle c-1's repair
    added = Matrix{Any}(undef, 0, 2 + size(X_train, 2))
    for c in 2:length(cycle_sizes)
        for r in (cycle_sizes[c - 1] + 1):cycle_sizes[c]
            added = vcat(added, permutedims(vcat(c - 1, final_y[r], final_X[r, :])))
        end
    end
    open(joinpath(OUTPUT_PATH, "added_$(name).csv"), "w") do io
        println(io, join(vcat("cycle", "label", ["V$(i)" for i in 1:size(X_train, 2)]), ","))
        writedlm(io, added, ',')
    end
    open(joinpath(OUTPUT_PATH, "probes_$(name).csv"), "w") do io
        println(io, "cycle,bits")
        for (c, bits) in enumerate(cycle_bits)
            println(io, "$(c - 1),$(bits)")
        end
    end

    certified = count(==('1'), last(cycle_bits))
    println(
        "$(name): $(length(cycle_bits) - 1) cycles, $(size(added, 1)) instances added, " *
        "$(certified)/$(size(probes, 1)) probes certified by the final model",
    )
    flush(stdout)
end

open(joinpath(OUTPUT_PATH, "train.csv"), "w") do io
    println(io, join(vcat("label", ["V$(i)" for i in 1:size(X_train, 2)]), ","))
    writedlm(io, hcat(y_train, X_train), ',')
end
open(joinpath(OUTPUT_PATH, "probes.csv"), "w") do io
    println(io, join(["V$(i)" for i in 1:size(X_train, 2)], ","))
    writedlm(io, probes, ',')
end

trace(true)
trace(false)
