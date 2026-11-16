# Compare how many instances each pipeline has to add before the model is certified:
# the original one (run_classification.sh, results in results/) and the one honouring the
# existing distribution (run_classification_honour.sh, results in results/honour).
#
# Execute using: julia --project=. experiments/compare_instances.jl <name> <modeltype>
#
# For every parameterization found in both result folders, it prints the mean number of
# added instances, repair cycles, F1 before/after, total time, and how much the variance
# of the free features (those outside the left constraint) drifted from the original
# training set at the end of the run.

using FormalVerificationRandomForests
using Printf
using Statistics
using SoleLogics
using SoleData

DATA_NAME = ARGS[1]
MODEL_TYPE = ARGS[2]
CONFIG_PATH = joinpath(pwd(), "experiments", "configurations")
include(joinpath(CONFIG_PATH, "configuration_$(DATA_NAME).jl"))

PIPELINES = [
    ("original", joinpath(pwd(), "results")),
    ("honour", joinpath(pwd(), "results", "honour")),
]

"""
Parse every repetition line of a result file into (numcycles, tottime, f1pre, f1post).
"""
function read_results(path)
    isfile(path) || return NamedTuple[]

    rows = NamedTuple[]
    for line in eachline(path)
        startswith(line, "(numtrees") || continue
        field(name) = parse(Float64, match(Regex("$(name) = ([^,)]+)"), line).captures[1])
        push!(
            rows,
            (
                numcycles=Int(field("numcycles")),
                tottime=field("tottime"),
                f1pre=field("metricpre"),
                f1post=field("metricpost"),
            ),
        )
    end

    return rows
end

"""
From one variance CSV, return the number of added instances and the median absolute
relative variance drift of the `free` features at the end of the run.
"""
function read_variance(path, free)
    lines = readlines(path)[2:end]
    isempty(lines) && return (added=0, drift=0.0)

    fields = [split(l, ",") for l in lines]
    ns = [parse(Int, f[1]) for f in fields]
    added = maximum(ns) - minimum(ns) + 1

    drift = Float64[]
    for f in fields
        parse(Int, f[1]) == maximum(ns) || continue
        feature = parse(Int, f[2])
        feature in free || continue
        new_variance, cumulative = parse(Float64, f[4]), parse(Float64, f[6])
        initial = new_variance - cumulative
        initial > 0 && push!(drift, abs(cumulative) / initial)
    end

    return (added=added, drift=isempty(drift) ? 0.0 : median(drift))
end

nfeatures = length(feature_cols(first(exp_datasets)))
_name = datasetname(first(exp_datasets))

@printf(
    "%-3s %-3s %-9s %5s %10s %8s %8s %8s %9s %12s\n",
    "LC", "HP", "pipeline", "reps", "instances", "cycles", "F1 pre", "F1 post",
    "time (s)", "free drift",
)

for lc in exp_lcomplexities, hp in exp_hparam[Symbol(MODEL_TYPE)], md in exp_maxdepth,
    bs in exp_batchsize, rc in exp_rcomplexities

    free = setdiff(1:nfeatures, constrained_indexes(exp_lconstraints[lc]))
    stem = "$(_name)_TY_$(MODEL_TYPE)_LC_$(lc)_RC_$(rc)_HP_$(hp)_MD_$(md)_BS_$(bs)"

    for (pipeline, root) in PIPELINES
        results = read_results(joinpath(root, "$(stem).txt"))
        isempty(results) && continue

        variances = [
            read_variance(joinpath(root, "variance", "$(stem)_REP_$(i).csv"), free) for
            i in 1:length(results) if
            isfile(joinpath(root, "variance", "$(stem)_REP_$(i).csv"))
        ]
        diverged = count(r -> r.numcycles < 0, results)

        @printf(
            "%-3d %-3d %-9s %5s %10s %8s %8.3f %8.3f %9.1f %12s\n",
            lc,
            hp,
            pipeline,
            diverged > 0 ? "$(length(results))*" : string(length(results)),
            isempty(variances) ? "-" : @sprintf("%.1f", mean(v.added for v in variances)),
            @sprintf("%.1f", mean(max(r.numcycles, 0) for r in results)),
            mean(r.f1pre for r in results),
            mean(r.f1post for r in results),
            mean(r.tottime for r in results),
            isempty(variances) ? "-" :
            @sprintf("%.1f%%", 100 * mean(v.drift for v in variances)),
        )
    end
end

println()
println("instances: mean number of instances added before certification")
println("free drift: median |variance change| / original variance over the free features")
println("* some repetitions hit the timeout instead of being certified")
