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

X, y = load_dataset(vehicle_config; root_path=DATA_PATH)
df = DataFrame(X, ["V$(i)" for i in 1:size(X)[2]])
plogiset = PropositionalLogiset(df);

# decisionlist = irepstar(plogiset, y)
# decisionList2 = sequentialcovering(plogiset, y, loss_function=ModalDecisionLists.LossFunctions.LaplaceMetric())
# decisionList3 = sequentialcovering(plogiset, y, loss_function=ModalDecisionLists.LossFunctions.Entropy())
# decisionList4 = sequentialcovering(plogiset, y, loss_function=ModalDecisionLists.LossFunctions.Entropy(), discretizedomain=true)
# decisionList5 = sequentialcovering(plogiset, y, loss_function=ModalDecisionLists.LossFunctions.LaplaceMetric(), discretizedomain=true)

# function called to train the actual base models in the list model
# model_wrapper(X, y, w; rng, iteration, kwargs...) = sequentialcovering(X, y, w, loss_function=ModalDecisionLists.LossFunctions.LaplaceMetric())
ensemble_model = build_ensemble(
    plogiset, y, 3; model_wrapper=sequentialcovering
)

# translate the ensemble to LRA format
my_lra = forest_decisionlist_to_lra(ensemble_model)
my_smtlib2 = to_smtlib2_classification(my_lra)

open("prova.smt2", "w") do f
    println(f, my_smtlib2)
end
