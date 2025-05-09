# ------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------

using Pkg
Pkg.add("DataFrames")
Pkg.add("Plots")
Pkg.add("StatsPlots")
Pkg.add("StatsBase")
Pkg.add("CSV")

include("73166321D_54157616E_48118254T_54152126Y.jl")
include("index.jl")  # Includes the index.jl file

using DataFrames
using Dates
using Plots
using StatsPlots
using Random
using StatsBase
using Statistics
using CSV
using DelimitedFiles
Random.seed!(12345)

# ------------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------------
# Path for the data file
data_path = "Entrega/optdigits.full"

# Function to load the data
function load_optdigits(filename)
    data = readdlm(filename, ',')
    inputs = convert(Matrix{Float32}, data[:, 1:64])
    targets = string.(convert(Vector{Int64}, data[:, 65]))
    return inputs, targets
end

# Load all data
println("Loading data...")
all_inputs, all_targets = load_optdigits(data_path)

# Create a dataframe for easier visualization
df_data = DataFrame(all_inputs, :auto)
df_data.target = all_targets

num_cols = size(df_data, 2)
feature_cols = num_cols - 1

# Create column names for the 64 features
new_names = [Symbol("x$i") for i in 1:feature_cols]
    
# Rename only the feature columns, leaving the last one (target) unchanged
for i in 1:feature_cols
    rename!(df_data, names(df_data)[i] => new_names[i])
end

# ------------------------------------------------------------------
# Train-Test Split using Cross-Validation Indices
# ------------------------------------------------------------------
println("Setting up train-test split using cross-validation indices...")

# Load cross-validation indices from the file (will be used as test set)
cv_indices_file = "Entrega/cv_indices.txt"
crossValidationIndices = vec(readdlm(cv_indices_file, Int64))

# Create train and test sets
test_indices = crossValidationIndices
train_indices = setdiff(1:size(all_inputs, 1), test_indices)

train_inputs = all_inputs[train_indices, :]
train_targets = all_targets[train_indices]

test_inputs = all_inputs[test_indices, :]
test_targets = all_targets[test_indices]

# Display basic information
println("Training set dimensions: ", size(train_inputs))
println("Test set dimensions: ", size(test_inputs))
println("Classes in training set: ", sort(unique(train_targets)))
println("Classes in test set: ", sort(unique(test_targets)))

# ------------------------------------------------------------------
# Exploratory Data Analysis
# ------------------------------------------------------------------
println("Generating visualizations...")
colors = [:lightblue, :red, :green, :yellow, :orange, :purple, :cyan, :magenta, :lightgreen, :brown]

# Class distribution in the training set
class_counts_train = countmap(train_targets)
class_labels_train = collect(keys(class_counts_train))
class_instances_train = collect(values(class_counts_train))
train_distribution = bar(class_labels_train, class_instances_train, color=colors,
                         legend=false, ylabel="Number of Instances",
                         xlabel="Digit", title="Digit Distribution in Training Set")

# Class distribution in the test set
class_counts_test = countmap(test_targets)
class_labels_test = collect(keys(class_counts_test))
class_instances_test = collect(values(class_counts_test))
test_distribution = bar(class_labels_test, class_instances_test, color=colors,
                       legend=false, ylabel="Number of Instances",
                       xlabel="Digit", title="Digit Distribution in Test Set")

# Histograms for selected features
selected_features = [1, 20, 40, 64]  # Selected features to visualize
histograms = [histogram(all_inputs[:, i], bins=20, title="Histogram of Feature $i", 
              label="Feature $i", legend=:outerright) for i in selected_features]

# Boxplots for selected features
boxplots = [boxplot(["Feature $i"], all_inputs[:, i], 
            title="Boxplot of Feature $i", label="Feature $i", 
            legend=:outerright) for i in selected_features]

# Correlation matrix for a sample of features
sample_features = 1:10  # First 10 features for the correlation matrix
cor_matrix = cor(all_inputs[:, sample_features])
hm = heatmap(cor_matrix, title = "Feature Correlation Matrix (Sample)",
        xticks = (1:length(sample_features), ["Feature $i" for i in sample_features]),
        yticks = (1:length(sample_features), ["Feature $i" for i in sample_features]),
        color = :coolwarm,
        size=(800,800))

# Visualize some digits as images
function plot_digit(features, index)
    # Reshape a 1x64 vector into an 8x8 matrix
    digit = reshape(features[index, :], (8, 8))
    # Transpose for correct visualization
    digit = permutedims(digit, (2, 1))
    # Visualize
    heatmap(digit, color=:grays, aspect_ratio=:equal, 
           title="Digit: $(all_targets[index])", yflip=true, 
           xticks=false, yticks=false)
end

# Visualize a sample of digits (one of each class)
sample_indices = [findfirst(x -> x == d, all_targets) for d in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]]
digit_plots = [plot_digit(all_inputs, i) for i in sample_indices]
digit_grid = plot(digit_plots..., layout=(2, 5), size=(800, 400), legend=false)

# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------
println("Normalizing data...")
# Calculate normalization parameters using only the training set
min_vals, max_vals = calculateMinMaxNormalizationParameters(train_inputs)

# Normalize both sets with the same parameters
train_inputs_norm = normalizeMinMax(train_inputs, (min_vals, max_vals))
test_inputs_norm = normalizeMinMax(test_inputs, (min_vals, max_vals))

# ------------------------------------------------------------------
# Model Setup
# ------------------------------------------------------------------
println("Setting up model configurations...")

# Define model configurations
topologies = [
    [15],
    [24],
    [32],
    [64],
    [2,5],
    [4,3],
    [10,10],
    [16, 16],
]

model_configurations = Dict(
    :ANN => [Dict("topology" => t, "numExecutions" => 50, "maxEpochs" => 100, "learningRate"=> 0.01, "validationRatio" =>0.2, "maxEpochsVal" => 20) for t in topologies],  # 8 neural network configurations
    :SVC => [Dict("kernel" => k, "C" => c, "gamma" => 0.1, "coef0" => 0.5, "degree" => 3) 
             for k in ["linear", "rbf", "poly", "sigmoid"] for c in [1, 10]],  # 8 SVM configurations
    :DoME => [Dict("maximumNodes" => n) for n in 5:12], # 8 node values
    :DecisionTreeClassifier => [Dict("max_depth" => d) for d in 7:12], # 6 depth values
    :KNeighborsClassifier => [Dict("n_neighbors" => k) for k in [1,3,5,7,9,11]]  # 6 neighbor values
)

# ------------------------------------------------------------------
# Model Training and Evaluation with K-Fold Cross-Validation
# ------------------------------------------------------------------
println("Training and evaluating models...")
all_results = Dict()
model_configuration_array = collect(pairs(model_configurations))

# Generate stratified k-fold indices for the training data
# We use k=5 for cross-validation
k = 5
training_cv_indices = crossvalidation(train_targets, k)

# Run cross-validation for each model configuration
for (modeltype, configs) in model_configuration_array
    println("Evaluating model: $modeltype")
    model_results = []
    for config in configs
        println("- Configuration: $config")
        result = modelCrossValidation(modeltype, config, (train_inputs_norm, train_targets), training_cv_indices)
        push!(model_results, (config, result))
    end
    all_results[modeltype] = model_results
end

# Create a dataframe with the results
println("Generating results...")
df_result = DataFrame(Model=String[], Params=Any[], Mean_Accuracy=Float64[], Std_Accuracy=Float64[])
for (modeltype, results) in all_results
    for result in results
        config = result[1]
        mean_acc = result[2][1][1]
        std_acc = result[2][1][2]
        push!(df_result, (String(modeltype), config, mean_acc, std_acc))
    end
end

# Sort by Mean Accuracy
sorted_df = sort(df_result, :Mean_Accuracy, rev=true)

# ------------------------------------------------------------------
# Final Evaluation on Test Set
# ------------------------------------------------------------------
println("Evaluating best model on test set...")

# Find the best model
best_model_row = sorted_df[1, :]
best_model_type = Symbol(best_model_row.Model)
best_model_config = best_model_row.Params

println("Best model: $best_model_type with configuration $best_model_config")

# Train the best model on the full training set
best_model = createModel(best_model_type, best_model_config)
trainModel!(best_model, (train_inputs_norm, train_targets))

# Evaluate on the test set
test_predictions = predict(best_model, test_inputs_norm)
test_accuracy = accuracy(test_predictions, test_targets)
println("Test set accuracy: $(round(test_accuracy * 100, digits=2))%")

# Create confusion matrix for test set
test_confusion_matrix = confusionMatrix(test_predictions, test_targets)

# ------------------------------------------------------------------
# Accuracy Comparison
# ------------------------------------------------------------------
println("Comparing model accuracies...")
best_configs = Dict()
for (modeltype, results) in all_results
    # Sort results for each model and take the best
    best_result = sort(results, by=x -> x[2][1][1], rev=true)[1]
    best_configs[modeltype] = best_result
end

# Get model types from Dict for plotting
model_types = [string(k) for k in keys(best_configs)]
# Extract accuracies
accuracies = [v[2][1][1] for v in values(best_configs)]

# Adjust plots for better visualization
min_accuracy, max_accuracy = minimum(accuracies), maximum(accuracies)
padding = (max_accuracy - min_accuracy) * 0.1
ylims_range = (min_accuracy - padding, max_accuracy + padding * 2)

acc_comparison = bar(model_types, accuracies, legend=false,
    ylabel="Accuracy (%)", xlabel="Model Type",
    title="Comparison of Best Model Accuracies",
    ylims=ylims_range,
    yticks=round(min_accuracy - padding, digits=3):0.005:round(max_accuracy + padding * 2, digits=3),
    bar_width=0.5,
    color=[:lightblue, :lightgreen, :lightcoral, :lightpink, :lightyellow],
    size=(800, 600))

# Add percentage to each bar
annotate!([(i, accuracies[i] + 0.002, text(string(round(accuracies[i] * 100, digits=2)) * "%", 10)) for i in 1:length(accuracies)])

# ------------------------------------------------------------------
# Confusion Matrix for the Best Model
# ------------------------------------------------------------------
println("Generating confusion matrix for the best model...")

# Create heatmap for the confusion matrix
confusion_heatmap = heatmap(test_confusion_matrix, 
                           title="Confusion Matrix for Best Model on Test Set",
                           xlabel="Predicted", ylabel="Actual",
                           xticks=(1:10, 0:9), yticks=(1:10, 0:9),
                           color=:blues,
                           size=(700, 600))

# Add values to each cell
for i in 1:size(test_confusion_matrix, 1)
    for j in 1:size(test_confusion_matrix, 2)
        if test_confusion_matrix[i, j] > 0
            annotate!([(j, i, text(string(test_confusion_matrix[i, j]), 8, :white))])
        end
    end
end

# ------------------------------------------------------------------
# Critical Difference Diagram
# ------------------------------------------------------------------
println("Generating critical difference diagram...")

# Extract results for all models for each fold
fold_results = Dict()

# Create a dictionary to store the best results by model type
best_models = Dict()
for (modeltype, results) in all_results
    # Find the best configuration for each model type
    best_config_index = argmax([r[2][1][1] for r in results])
    best_models[modeltype] = results[best_config_index]
    
    # Extract results by fold for the best configuration
    # In results, [2][5] is the vector with accuracy for each fold
    fold_results[modeltype] = best_models[modeltype][2][5]
end

# Convert to matrix for CD diagram
# Each row is a fold/dataset, each column is a method
methods = [string(key) for key in keys(fold_results)]
n_folds = length(first(values(fold_results)))
performances = zeros(n_folds, length(methods))

for (i, method) in enumerate(methods)
    performances[:, i] = fold_results[Symbol(method)]
end

# Create CD diagram
cd_diagram = create_cd_diagram(
    methods, 
    performances, 
    α=0.05, 
    lower_is_better=false,  # We use accuracy, so higher values are better
    title="ML Model Comparison",
    figsize=(900, 500)
)

display(cd_diagram)

# ------------------------------------------------------------------
# Save Results
# ------------------------------------------------------------------
println("Saving results...")

# Create directory for plots if it doesn't exist
mkpath("plots")

# Save results dataframe
CSV.write("results_optdigits.csv", sorted_df)

# Save plots
savefig(train_distribution, "plots/train_distribution.png")
savefig(test_distribution, "plots/test_distribution.png")
savefig(hm, "plots/correlation_heatmap.png")
savefig(acc_comparison, "plots/accuracy_comparison.png")
savefig(confusion_heatmap, "plots/confusion_matrix.png")
savefig(digit_grid, "plots/digit_samples.png")

# Save CD diagram
savefig(cd_diagram, "plots/cd_diagram.png")

# Save selected histograms
for (i, idx) in enumerate(selected_features)
    savefig(histograms[i], "plots/histogram_feature$(idx).png")
end

# Save selected boxplots
for (i, idx) in enumerate(selected_features)
    savefig(boxplots[i], "plots/boxplot_feature$(idx).png")
end

println("Analysis completed and results saved.")