using FormalVerificationRandomForests

using DataFrames
using DecisionTree
using JuMP
using Ipopt
using Random
using StatsBase
using Statistics: cor

using SoleModels
using ModalDecisionLists

using MLJ, MLJXGBoostInterface
import MLJ: @load
using XGBoost

OUTPUT_PATH = joinpath(pwd(), "results")
DATA_PATH = joinpath(pwd(), "dataset")

iris_config = DatasetConfig(
    "iris",
    "iris.csv",
    ',';
    skipstart=1,
    target_col=5,
    feature_cols=1:4,
    dataset_type=0,
    mappings=Dict(
        5 => Dict(
            "Iris-setosa" => 1,
            "Iris-versicolor" => 2,
            "Iris-virginica" => 3,
        ),
    ),
)

X, y = load_dataset(iris_config; root_path=DATA_PATH)

df = DataFrame(X, ["V$(i)" for i in 1:size(X, 2)])
df.y = categorical(y)

plogiset = PropositionalLogiset(df);

model = XGBoostClassifier(; num_round=1)

# features = all columns except :y, target = :y
m = machine(model, df[:, 1:(end - 1)], df.y)
MLJ.fit!(m)

trees = XGBoost.trees(m.fitresult[1])
featurenames = m.report.vals[1].features
classlabels = String.(levels(df.y))
my_model = solemodel(trees, Matrix(X), String.(y); classlabels)

preds = apply!(my_model, df[:, 1:(end - 1)], String.(y))

# to see the weights associated to the leafs, do:
# my_model.info.leaf_value (or my_model.models[i] and then .info.leaf_value)

# X = scalarlogiset(df[:, 1:(end-1);
#     relations=AbstractRelation[],
#     conditions=Vector{ScalarMetaCondition}(
#         collect(Iterators.flatten([
#             [ScalarMetaCondition(f, <=) for f in VariableValue.(1:4)],
#             [ScalarMetaCondition(f, >=) for f in VariableValue.(1:4)]
#         ]))
#     )
# )

# TODO: of course, we need to implement the LRA translation
my_lra = xgboost_to_lra(my_model)
my_smtlib2 = to_smtlib2_classification_xgboost(my_lra)

open("prova_XGBOOST.smt2", "w") do f
    println(f, my_smtlib2)
end
