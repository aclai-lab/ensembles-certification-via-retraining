"""
    experimentmodel, dataset, constraints; my_smt=:z3, opt_solver=:gurobi)

Implements the pipeline for formal verification and certificate the decision forest models.
This function iteratively verifies constraints and repairs violations until convergence.

The function performs verification and repair through the following iterative steps:
1. Converts the decision forest model into Linear Real Arithmetic (LRA) format.
2. Uses an SMT solver to find counterfactual explanations violating constraints.
3. If a counterfactual is found, repairs by adding it to the dataset and retraining.
The process continues until no further counterexamples can be found.
"""
function experiment(
    # could be provided by the called; not a big deal, loading time is ~0.02s for "abalone"
    X,
    y,
    config,
    ::Val{:classification};
    smt2filename::String="$(smt2filename)",

    # model type selector;
    # this can be, :decision_forest, :list_forest, :xgboost
    model_type::Symbol=:decision_forest,

    # stuff related to decision forests
    n_subfeatures::Int=-1,
    n_trees::Int=1,
    partial_sampling::Float64=1.0,
    forest_max_depth::Int=-1,

    # stuff related to lists 
    n_lists::Int=1,

    # stuff related to xgboost
    n_rounds::Int=1,
    pure_lra::Bool=false,
    lconstraint_complexity::Int=1,
    default_lconstraints::Vector{SyntaxBranch}=SyntaxBranch[],
    rconstraint_complexity::Int=1,
    default_rconstraints::Vector{SyntaxBranch}=SyntaxBranch[],
    batch_size::Int=1,
    M::Int=100000,
    TIMEOUT::Float64=3000.0, # timeout after which the experiment is declared as divergent
    silent::Bool=false,
    # if set, per-feature variance changes are also written to this CSV file
    variance_csv::Union{Nothing,String}=nothing,
    # if true, the features outside the left constraint may only take the values of a
    # real training instance (both in the SMT query and in the repair step); if false,
    # the original pipeline runs, where z3 is free to pick any value in their range
    honour_distribution::Bool=false,
    # only with honour_distribution: number of real profiles added per counterexample (the counterexample
    # itself included), see `nearest_profiles`
    profiles_per_cycle::Int=5,
    # if set, called as on_cycle(numcycles, model, X, y) right after each model is
    # trained, e.g. to record how certification evolves
    on_cycle::Union{Nothing,Function}=nothing,
    # e.g., to forward `root_path` in `load_dataset`
    kwargs...,
)
    convergence_time, numcycles = 0.0, 0

    # we want to keep track of the first model that is trained,
    # since we will compare the performance of this with the last (certified) one
    first_model_trained = nothing

    # pick the correct left and right constraint
    lconstraint = default_lconstraints[lconstraint_complexity]
    rconstraint = default_rconstraints[rconstraint_complexity]

    @info lconstraint
    @info rconstraint

    # this will be useful later; see the Counterexample Repair section
    rconstraint_dnf = dnf(rconstraint)

    !silent && printstyled("Boundary constraint generation...\n"; color=:green)
    boundary_constraints = make_boundary_constraint(X)

    # this is the final constraint
    constraint = IMPLICATION(lconstraint, rconstraint)

    if istrivally_unsat(
        config, lconstraint, rconstraint, boundary_constraints, smt2filename
    )
        return nothing
    end

    # the features named in the left constraint are the ones the certification moves;
    # every other (free) feature may only take a profile of a real training instance,
    # so that the counterexamples honour the existing distribution
    constrained = constrained_indexes(lconstraint)
    free = setdiff(1:size(X, 2), constrained)
    domain_assertion = if honour_distribution
        free_profile_domain(X, free; integerfeatures=integerfeatures(config))
    else
        ""
    end
    original_X = copy(X)
    profile_scale = [s == 0 ? 1.0 : s for s in vec(std(X[:, free]; dims=1))]

    # running per-feature variance of X, updated each time an instance is added
    variance_state = WelfordState(X)
    initial_variance = welford_variance(variance_state)
    if !isnothing(variance_csv)
        mkpath(dirname(variance_csv))
        write(variance_csv, VARIANCE_CSV_HEADER * "\n")
    end

    while true
        !silent && printstyled("Creating (or cleaning) optimizators...")
        jump_model = make_optimizer()

        !silent && printstyled(
            "Forest training & SoleModel conversion...\n"; color=:green
        )

        # train depending on the type requested by the experiments runner
        model = nothing
        if model_type == :decision_forest
            model = build_forest(y, X, -1, n_trees, 1.0, forest_max_depth, 5, 2)
        elseif model_type == :list_forest
            # adapt X, y to match ModalDecisionLists.jl
            df = DataFrame(X, ["V$(i)" for i in 1:size(X)[2]])
            plogiset = PropositionalLogiset(df)

            # notice how the correct hyperparameter is specified (n_lists)
            model = build_ensemble(
                plogiset, y, n_lists; model_wrapper=sequentialcovering
            )
        elseif model_type == :xgboost
            # this part is copy-pasted from the XGBoost examples of SoleXplorer
            df = DataFrame(X, ["V$(i)" for i in 1:size(X)[2]])
            df.y = categorical(y)

            plogiset = PropositionalLogiset(df)
            model = XGBoostClassifier(;
                num_round=n_rounds, max_depth=10, tree_method="exact"
            )
            m = machine(model, df[:, 1:(end - 1)], df.y)
            MLJ.fit!(m)

            trees = XGBoost.trees(m.fitresult[1])
            classlabels = String.(levels(df.y))

            model = solemodel(
                trees, Matrix(X), String.(y); classlabels=classlabels
            )
        else
            println("Error! Invalid model_type provided!")
            exit(3)
        end

        smodel = solemodel(model)

        !isnothing(on_cycle) && on_cycle(numcycles, model, X, y)

        if isnothing(first_model_trained)
            first_model_trained = deepcopy(model)
        end

        !silent &&
            printstyled("Converting the model to LRA format...\n"; color=:green)

        lra = nothing
        if model_type == :decision_forest
            lra = forest_to_lra(smodel; pure_lra=pure_lra)
        elseif model_type == :list_forest
            lra = forest_decisionlist_to_lra(smodel; pure_lra=pure_lra)
        elseif model_type == :xgboost
            lra = xgboost_to_lra(smodel; pure_lra=pure_lra)
        else
            println("Error! Invalid model_type provided!")
            exit(4)
        end

        lra_to_evaluate = CONJUNCTION(
            lra, NEGATION(constraint), boundary_constraints...
        )

        !silent && printstyled(
            "To smt2 output file ($(smt2filename))...\n"; color=:green
        )

        smtlib2_translator = nothing
        if model_type == :decision_forest || model_type == :list_forest
            smtlib2_translator = to_smtlib2_classification
        elseif model_type == :xgboost
            smtlib2_translator = to_smtlib2_classification_xgboost
        else
            println("Error! Invalid model_type provided!")
            exit(4)
        end

        smtlib2_translator(
            lra_to_evaluate;
            output_file=smt2filename,
            force_input=["Or"],
            integerfeatures=integerfeatures(config),
            typeof_output=dataset_type(config),
        )
        honour_distribution && inject_smt_assertion(smt2filename, domain_assertion)

        # os is the value of the "big O" (e.g., Or in the case of regression)
        vs, os, O = nothing, nothing, nothing

        !silent && printstyled("Formal verification...\n"; color=:green)
        stats = @timed begin
            # call z3... is the smt2 file unsatisfiable? If it is not the case, go on
            _verify_res = verify(smt2filename)
            if _verify_res == false
                # same 7-tuple shape as the regression pipeline, which is what
                # experiments/run_experiment.jl unpacks; classification has no
                # adhoc (MLJ) models, so those two slots are `nothing`
                return (
                    model,
                    nothing,
                    convergence_time,
                    numcycles,
                    IMPLICATION(lconstraint, rconstraint),
                    first_model_trained,
                    nothing,
                )
            else
                vs, os = _verify_res # counterexample
            end

            O = adjustement_of_counterexample(
                config, jump_model, os, rconstraint_dnf, M; output_name="Or"
            )
            if isnothing(O)
                return nothing
            end
        end

        O = string(convert(Int, round(O)))

        # finally, add a new (and eventually perturbed) instance to the dataset
        new_x = [
            parse(Float64, replace(strip(last(split(v, "="))), " " => "")) for
            v in vs
        ]

        function add_instance!(instance)
            X = vcat(X, instance')
            y = vcat(y, O)

            old_variance = welford_variance(variance_state)
            welford_update!(variance_state, instance)
            log_variance_change(
                variance_csv,
                variance_state.n,
                old_variance,
                welford_variance(variance_state),
                initial_variance,
            )
        end

        for _ in 1:batch_size
            _p = perturb(new_x, lconstraint)
            add_instance!(batch_size == 1 ? new_x : _p)
        end

        # the counterexample's free part is one real profile; also add its nearest real
        # profiles, moved into the constraint, so that a forest leaf can actually flip
        # (one instance alone is below the minimum leaf size)
        _ngrafts = honour_distribution ? profiles_per_cycle - 1 : 0
        for row in nearest_profiles(
            original_X, free, new_x, _ngrafts; scale=profile_scale
        )
            grafted = copy(original_X[row, :])
            grafted[constrained] = new_x[constrained]
            add_instance!(grafted)
        end

        @info "Repeat cycle and add the $(length(y))-th instance to the dataset"
        convergence_time += (stats.time - stats.compile_time) * batch_size
        numcycles += 1 * batch_size

        # divergence condition
        if convergence_time > TIMEOUT
            return (
                model,
                nothing,
                convergence_time,
                -1,
                IMPLICATION(lconstraint, rconstraint),
                first_model_trained,
                nothing,
            )
        end
    end
end
