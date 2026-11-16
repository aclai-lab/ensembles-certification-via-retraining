# This file contains specific dispatches that are defined in
# SoleModels.jl, at ext/XGBoostExt.jl

using SoleModels
using XGBoost
using CategoricalArrays

import SoleModels: alphabet, solemodel

# ---------------------------------------------------------------------------- #
#                          DecisionXGBoost alphabet                            #
# ---------------------------------------------------------------------------- #
function alphabet(model::XGBoost.Booster; kwargs...)
    function _alphabet!(a::Vector, tree::XGBoost.Node; kwargs...)
        # Base case: leaf node has no split
        if isnothing(tree.split) || length(tree.children) == 0
            return a
        end

        feature = Sole.VariableValue(
            tree.split isa String ? Symbol(tree.split) : tree.split
        )
        condition = ScalarCondition(feature, (<), tree.split_condition)
        push!(a, condition)

        if length(tree.children) == 2
            _alphabet!(a, tree.children[1]; kwargs...)
            _alphabet!(a, tree.children[2]; kwargs...)
        else
            error(
                "Found $(length(tree.children)) children while 2 were expected: $(tree.children).",
            )
        end
        return a
    end

    # FIX: the Booster overload was calling _alphabet! on the Booster itself
    # (infinite recursion risk) instead of iterating its trees. Now we iterate
    # over the dumped trees properly.
    trees = XGBoost.trees(model)
    a = Atom{ScalarCondition}[]
    for t in trees
        _alphabet!(a, t; kwargs...)
    end
    return a
end

# ---------------------------------------------------------------------------- #
#                              Helper utilities                                 #
# ---------------------------------------------------------------------------- #

"""
    get_condition(featidstr, featval, featurenames; test_operator)

Build a ScalarCondition from an XGBoost feature-id string (e.g. "f0", "f1", …).
XGBoost uses 0-based indexing, so we add 1 to get the Julia column index.
"""
function get_condition(featidstr::AbstractString, featval, featurenames; test_operator)
    # FIX: guard against unexpected feature-id formats
    featid = try
        parse(Int, featidstr[2:end]) + 1   # "f0" -> 1, "f1" -> 2, …
    catch
        error("Cannot parse XGBoost feature id: $featidstr")
    end
    feature = if isnothing(featurenames)
        VariableValue(featid)
    else
        VariableValue(featid, featurenames[featid])
    end
    return ScalarCondition(feature, test_operator, featval)
end

"""
    get_condition(class_idx, featurenames; test_operator, featval)

Build a ScalarCondition from a class index (used for single-node trees).
"""
function get_condition(class_idx::Int, featurenames; test_operator, featval)
    feature = if isnothing(featurenames)
        VariableValue(class_idx)
    else
        VariableValue(class_idx, featurenames[class_idx])
    end
    return ScalarCondition(feature, test_operator, featval)
end

get_operator(atom::Atom{<:ScalarCondition})     = atom.value.metacond.test_operator
get_i_variable(atom::Atom{<:ScalarCondition})   = atom.value.metacond.feature.i_variable
get_threshold(atom::Atom{<:ScalarCondition})    = atom.value.threshold

"""
    satisfies_conditions(row, formula)

Return `true` iff `row` satisfies every atom in `formula`.

FIX: an empty `formula` previously returned `true` for every row (vacuous truth),
making all instances appear covered by every path. We now treat an empty path as
"no constraint" only when that is the *intended* semantic (i.e. the root of a
single-node tree). Callers that build paths incrementally always have at least
one atom before reaching a leaf, so this is safe.
"""
function satisfies_conditions(row, formula)
    isempty(formula) && return true   # root-level / unconstrained leaf: explicit and intentional
    return all(
        atom -> get_operator(atom)(row[get_i_variable(atom)], get_threshold(atom)),
        formula,
    )
end

"""
    bitmap_check_conditions(X, formula)

Return a `BitVector` of length `nrow(X)` indicating which rows satisfy all
conditions in `formula`.
"""
function bitmap_check_conditions(X, formula)
    return BitVector([satisfies_conditions(row, formula) for row in eachrow(X)])
end

# ---------------------------------------------------------------------------- #
#                    early_return (single-node tree helper)                    #
# ---------------------------------------------------------------------------- #

"""
    early_return(leaf, antecedent, clabel, classl)

Build a trivial Branch for a tree that consists of a single leaf node (no split).
Both children point to the same ConstantModel since the condition is vacuously true.

FIX: removed the forced `string(...)` conversion on `label`, which was silently
coercing numeric regression targets into strings and breaking regressor output.
We preserve the original type and only fall back to String for classifiers
(where `classlabels` is provided).
"""
function early_return(leaf, antecedent, clabel, classl)
    # For classifiers clabel / classl are already categorical / string-like;
    # for regressors they are nothing — keep the numeric leaf value instead.
    label = if isnothing(clabel)
        leaf   # regression: use the raw leaf score
    else
        clabel isa AbstractVector ? first(clabel) : clabel
    end

    # supporting_predictions and supporting_labels must be parallel per-instance
    # vectors of equal length. SoleModels' evaluate.jl calls e.g. mse(labels, preds)
    # and asserts length equality. We store one entry per covered instance;
    # for a single-node tree that covers everything we use a length-1 pair.
    info = (;
        leaf_value             = leaf,
        supporting_predictions = isnothing(label)  ? eltype(y)[]     : [label],
        supporting_labels      = isnothing(classl) ? eltype(y)[]     : [classl],
    )

    return Branch(
        antecedent,
        SoleModels.ConstantModel(label, info),
        SoleModels.ConstantModel(label, info),
        info,
    )
end

# ---------------------------------------------------------------------------- #
#                       solemodel — Vector{XGBoost.Node}                       #
# ---------------------------------------------------------------------------- #

function SoleModels.solemodel(
    model::Vector{<:XGBoost.Node},
    X::AbstractMatrix,
    y::AbstractVector;
    classlabels  = nothing,
    featurenames = nothing,
    keep_condensed::Bool = false,
    use_float32::Bool    = true,
    kwargs...,
)
    keep_condensed && error("Cannot keep condensed XGBoost.Node.")

    isnothing(classlabels) || (nclasses = length(classlabels))

    trees = map(enumerate(model)) do (i, t)
        if isnothing(classlabels)
            class_idx = nothing
            clabels   = nothing
        else
            class_idx = (i - 1) % nclasses + 1
            clabels   = categorical([classlabels[class_idx]])
        end

        # Single-node tree (no split at all)
        if isnothing(t.split)
            antecedent = if isnothing(class_idx)
                # Regression: dummy condition that is always true
                Atom(ScalarCondition(VariableValue(1), (<), Inf))
            else
                Atom(
                    get_condition(
                        class_idx, featurenames; test_operator = (<), featval = Inf
                    ),
                )
            end
            leaf = use_float32 ? Float32(t.leaf) : t.leaf
            early_return(leaf, antecedent, clabels, isnothing(classlabels) ? nothing : classlabels[class_idx])
        else
            SoleModels.solemodel(
                t,
                X,
                y;
                classlabels,
                class_idx,
                clabels,
                featurenames,
                use_float32,
                kwargs...,
            )
        end
    end

    # FIX: filter out `nothing` leaves that were returned when no instance
    # reached a branch (instead of the old `trues` fallback).
    trees = filter(!isnothing, trees)

    info = merge(
        isnothing(featurenames) ? (;) : (; featurenames = featurenames),
        (;
            leaf_value = reduce(
                vcat, getindex.(getproperty.(trees, :info), :leaf_value)
            ),
            supporting_predictions = reduce(
                vcat,
                getindex.(getproperty.(trees, :info), :supporting_predictions),
            ),
            supporting_labels = reduce(
                vcat, getindex.(getproperty.(trees, :info), :supporting_labels)
            ),
        ),
    )

    return DecisionXGBoost(trees, info)
end

# ---------------------------------------------------------------------------- #
#                       solemodel — single XGBoost.Node                       #
# ---------------------------------------------------------------------------- #

"""
    solemodel(tree::XGBoost.Node, X, y; ...)

Recursively convert one XGBoost tree (a tree of `XGBoost.Node`s) into a
`Branch` / `ConstantModel` structure compatible with SoleModels.

Key invariant: `path_conditions` contains exactly the atoms on the path from
the root to the *current* node, so a leaf receives the complete, correct set
of conditions that must hold for an instance to reach it.
"""
function SoleModels.solemodel(
    tree::XGBoost.Node,
    X::AbstractMatrix,
    y::AbstractVector;
    classlabels      = nothing,
    class_idx        = nothing,
    clabels          = nothing,
    featurenames     = nothing,
    path_conditions  = Formula[],   # atoms accumulated from root to here
    use_float32::Bool = true,
)
    split_condition = use_float32 ? Float32(tree.split_condition) : tree.split_condition

    antecedent = Atom(
        get_condition(tree.split, split_condition, featurenames; test_operator = (<))
    )

    # Left branch: condition is  feature < threshold
    left_path = copy(path_conditions)
    push!(
        left_path,
        Atom(get_condition(tree.split, split_condition, featurenames; test_operator = (<))),
    )

    # Right branch: condition is  feature ≥ threshold
    right_path = copy(path_conditions)
    push!(
        right_path,
        Atom(get_condition(tree.split, split_condition, featurenames; test_operator = (≥))),
    )

    lefttree = if isnothing(tree.children[1].split)
        xgbleaf(
            tree.children[1];
            formula = left_path,
            X, y, classlabels, use_float32,
        )
    else
        SoleModels.solemodel(
            tree.children[1],
            X, y;
            path_conditions = left_path,
            classlabels, class_idx, clabels, featurenames, use_float32,
        )
    end

    righttree = if isnothing(tree.children[2].split)
        xgbleaf(
            tree.children[2];
            formula = right_path,
            X, y, classlabels, use_float32,
        )
    else
        SoleModels.solemodel(
            tree.children[2],
            X, y;
            path_conditions = right_path,
            classlabels, class_idx, clabels, featurenames, use_float32,
        )
    end

    # FIX: if either child returned `nothing` (empty-support leaf), propagate
    # `nothing` upward so the caller can prune this branch entirely.
    (isnothing(lefttree) || isnothing(righttree)) && return nothing

    info = (;
        leaf_value = [
            lefttree.info[:leaf_value]...,
            righttree.info[:leaf_value]...,
        ],
        supporting_predictions = [
            lefttree.info[:supporting_predictions]...,
            righttree.info[:supporting_predictions]...,
        ],
        supporting_labels = [
            lefttree.info[:supporting_labels]...,
            righttree.info[:supporting_labels]...,
        ],
    )

    return Branch(antecedent, lefttree, righttree, info)
end

# ---------------------------------------------------------------------------- #
#                                  xgbleaf                                     #
# ---------------------------------------------------------------------------- #

"""
    xgbleaf(leaf; formula, X, y, classlabels, use_float32)

Build a `ConstantModel` for a leaf node.

### `supporting_labels` / `supporting_predictions` contract

SoleModels' `evaluate.jl` (e.g. `mse`, `accuracy`) asserts that these two
vectors have the **same length** and treats them as parallel per-instance
arrays: `supporting_labels[i]` is the ground-truth value and
`supporting_predictions[i]` is the model's prediction for the i-th covered
instance.

We therefore store:
- `supporting_labels`      = `y[bitX]`                  (one entry per covered row)
- `supporting_predictions` = `fill(prediction, n_covered)` (repeated scalar)

Fixes applied
─────────────
1. **No more `trues` fallback.** Dead leaves (no training instance reaches
   them) return `nothing`; callers prune these branches.
2. **Parallel vectors of correct length**, not `unique(y)` vs a single scalar.
3. **Regression dead-leaf guard** added (previously regression never checked).
"""
function xgbleaf(
    leaf::XGBoost.Node;
    formula::Vector{<:Formula},
    X::AbstractMatrix,
    y::AbstractVector,
    classlabels = nothing,
    use_float32::Bool,
)
    leaf_value = use_float32 ? Float32(leaf.leaf) : leaf.leaf

    if isnothing(classlabels)
        # ── Regression ────────────────────────────────────────────────────────
        # supporting_predictions and supporting_labels must be parallel vectors
        # of the same length: one entry per instance that reaches this leaf.
        # SoleModels' evaluate.jl asserts length equality before computing mse.
        bitX = bitmap_check_conditions(X, formula)

        if !any(bitX)
            # Dead leaf: no instance reaches here. Return nothing so the caller
            # can prune this branch rather than corrupt coverage stats.
            return nothing
        end

        covered_y = y[bitX]          # ground-truth values of covered instances
        n_covered  = length(covered_y)

        info = (;
            leaf_value,
            # Repeat the single leaf prediction once per covered instance so
            # that zip(supporting_labels, supporting_predictions) is well-defined.
            supporting_predictions = fill(leaf_value, n_covered),
            supporting_labels      = covered_y,          # per-instance ground truth
        )
        return SoleModels.ConstantModel(leaf_value, info)
    else
        # ── Classification ────────────────────────────────────────────────────
        bitX = bitmap_check_conditions(X, formula)

        # If no instance reaches this leaf, return nothing so the caller prunes
        # this branch (avoids the old `trues` fallback that inflated coverage).
        !any(bitX) && return nothing

        covered_y  = y[bitX]
        n_covered  = length(covered_y)

        prediction = SoleModels.bestguess(covered_y; suppress_parity_warning = true)
        isnothing(prediction) && return nothing

        info = (;
            leaf_value,
            # One prediction per covered instance (same value repeated) so that
            # supporting_predictions and supporting_labels are parallel vectors.
            supporting_predictions = fill(prediction, n_covered),
            supporting_labels      = covered_y,          # per-instance ground truth
        )
        return SoleModels.ConstantModel(prediction, info)
    end
end
