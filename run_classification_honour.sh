#!/usr/bin/bash

# Same as run_classification.sh, but runs experiments/run_experiment_honour.jl, whose
# counterexamples only use real profiles for the features outside the left constraint.
# Results go to results/honour; compare them with results/ using
# julia --project=. experiments/compare_instances.jl

NAMES=(
    breastcancer
)

MODELS=(
    decision_forest
)

for name in "${NAMES[@]}"; do
    for model in "${MODELS[@]}"; do
        SESSION="exp_${name}_${model}_honour"

        echo "Starting tmux session: $SESSION"

        tmux new-session -d -s "$SESSION"

        tmux send-keys -t "$SESSION" \
            "julia --project=. experiments/run_experiment_honour.jl ${name} ${model}" C-m
    done
done

echo "All experiments started."
echo "Attach with: tmux attach -t exp_<name>_<model>_honour"
