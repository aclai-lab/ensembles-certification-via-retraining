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

include("../experiments/utils.jl")

OUTPUT_PATH = joinpath(pwd(), "results")
DATA_PATH = joinpath(pwd(), "dataset")

vehicle_config = DatasetConfig(
    "vehicle",
    "vehicle.csv",
    ',';
    skipstart=1,
    target_col=3,
    feature_cols=Iterators.flatten((2:2, 4:5)),
    dataset_type=1,
)

# copia incolla da un altro flie
X, y = load_dataset(vehicle_config; root_path=DATA_PATH)
df = DataFrame(X, ["V$(i)" for i in 1:size(X)[2]])
df.y = y
plogiset = PropositionalLogiset(df);

# quello che dobbiamo fare
# X, y = load_dataset(iris_config; root_path=DATA_PATH)
# df = DataFrame(X, ["V$(i)" for i in 1:size(X, 2)])
# df.y = categorical(y)
# plogiset = PropositionalLogiset(df);

model = XGBoostRegressor(; num_round=3)

# features = all columns except :y, target = :y
m = machine(model, df[:, 1:(end - 1)], df.y)
MLJ.fit!(m)

_base_score = mean(y) # m.model.base_score

trees = XGBoost.trees(m.fitresult[1])
featurenames = m.report.vals[1].features
# classlabels = String.(levels(df.y))
my_model = solemodel(trees, Matrix(X), y)


# The apply for XGBoostRgressors is not defined in SoleModels.
# # NEW APPLY
# apply(
#     my_model,
#     SoleData.scalarlogiset(
#         DataFrame(X, ["V$(i)" for i in 1:size(X)[2]]),
#         allow_propositional=true,
#     ),
# )
#
# # OLD APPLY
# apply(my_model, plogiset)

preds = apply!(my_model, df[:, 1:(end - 1)], y)
my_lra = FormalVerificationRandomForests.xgboost_to_lra(
    my_model; base_score=_base_score
)

to_smtlib2(
    my_lra;
    output_file="prova_XGBOOST.smt2",
    # output_file=smt2filename,
    force_input=["Or"],
    # integerfeatures=integerfeatures(config),
    # typeof_output=dataset_type(config),
)

# open("prova_XGBOOST.smt2", "w") do f
#     println(f, my_smtlib2)
# end
