# Same experiments as run_experiment.jl, but the counterexamples honour the existing
# distribution: every feature outside the left constraint takes the values of a real
# training instance, and each repair cycle also adds the nearest real profiles moved into
# the constraint. Results are written to results/honour.
#
# Execute using: julia --project=. experiments/run_experiment_honour.jl <name> <modeltype>
HONOUR_DISTRIBUTION = true

include("run_experiment.jl")
