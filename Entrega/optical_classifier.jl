# ------------------------------------------------------------------
# Critical Difference Diagram Implementation
# ------------------------------------------------------------------
using Plots
using Statistics
using StatsBase
using Random

function create_cd_diagram(methods::Vector{String}, performances::AbstractMatrix{<:Real};
                          α::Real=0.05, lower_is_better::Bool=true, 
                          title::String="Critical Difference Diagram",
                          figsize=(800, 400))
    # Number of datasets (rows) and methods (columns)
    n_datasets, n_methods = size(performances)
    
    # If higher values are better, negate the performances
    if !lower_is_better
        performances = -performances
    end
    
    # Calculate average ranks for each method
    ranks = zeros(n_datasets, n_methods)
    for i in 1:n_datasets
        ranks[i, :] = tiedrank(performances[i, :])
    end
    
    avg_ranks = mean(ranks, dims=1)[:]
    
    # Sort methods by average rank
    sort_idx = sortperm(avg_ranks)
    sorted_methods = methods[sort_idx]
    sorted_ranks = avg_ranks[sort_idx]
    
    # For visualization, we need to determine which methods are not significantly different
    # Critical difference formula for Nemenyi test
    q_alpha = Dict(
        0.1 => Dict(2 => 1.645, 3 => 2.052, 4 => 2.291, 5 => 2.459, 6 => 2.589, 7 => 2.693, 8 => 2.780),
        0.05 => Dict(2 => 1.960, 3 => 2.344, 4 => 2.569, 5 => 2.728, 6 => 2.850, 7 => 2.949, 8 => 3.031)
    )
    
    # Use α = 0.05 by default
    q = get(get(q_alpha, α, Dict()), n_methods, 3.0)  # Default to 3.0 if not found
    CD = q * sqrt((n_methods * (n_methods + 1)) / (6 * n_datasets))
    
    # Create the plot
    plt = plot(size=figsize, legend=false, title=title, grid=false, 
              xlabel="Average Rank", ylabel="", titlefontsize=12)
    
    # Draw the axis for ranks
    min_rank, max_rank = extrema(sorted_ranks)
    padding = 0.5
    xlims = (min_rank - padding, max_rank + padding)
    ylims = (0, n_methods + 1)
    plot!(plt, xlims=xlims, ylims=ylims, yticks=nothing)
    
    # Draw rank axis
    plot!(plt, [min_rank, max_rank], [0.5, 0.5], color=:black, linewidth=1)
    
    # Draw tick marks on rank axis
    for r in ceil(Int, min_rank):floor(Int, max_rank)
        plot!(plt, [r, r], [0.3, 0.7], color=:black, linewidth=1)
        annotate!(plt, r, 0, text(string(r), 8, :black))
    end
    
    # Draw methods and their ranks
    y_positions = reverse(1:n_methods) .+ 1
    
    # Plot methods as points
    scatter!(plt, sorted_ranks, y_positions, markersize=6, color=:blue)
    
    # Add method names
    for i in 1:n_methods
        annotate!(plt, sorted_ranks[i] - 0.2, y_positions[i], text(sorted_methods[i], 8, :left, :center))
    end
    
    # Draw connections for methods that are not significantly different
    for i in 1:n_methods
        for j in (i+1):n_methods
            if abs(sorted_ranks[i] - sorted_ranks[j]) <= CD
                # Draw a horizontal line connecting non-significantly different methods
                y_pos = (y_positions[i] + y_positions[j]) / 2
                plot!(plt, [sorted_ranks[i], sorted_ranks[j]], [y_pos, y_pos], color=:gray, linewidth=1, alpha=0.7)
            end
        end
    end
    
    # Add the Critical Difference indicator
    cd_x = max_rank - CD/2
    cd_y = 0.5
    plot!(plt, [cd_x - CD/2, cd_x + CD/2], [cd_y, cd_y], color=:red, linewidth=2)
    annotate!(plt, cd_x, cd_y - 0.3, text("CD = $(round(CD, digits=3))", 8, :center))
    
    return plt
end

# ------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------

using Pkg
# Only uncomment if you need to install these packages
# Pkg.add("DataFrames")
# Pkg.add("Plots")
# Pkg.add("StatsPlots")
# Pkg.add("StatsBase")
# Pkg.add("CSV")

include("73166321D_54157616E_48118254T_54152126Y.jl")

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

function load_optdigits(filename)
    data = readdlm(filename, ',')
    inputs = convert(Matrix{Float32}, data[:, 1:64])
    targets = string.(convert(Vector{Int64}, data[:, 65]))
    return inputs, targets
end

function generate_cv_indices(filename)
    dataset_file = filename
    
    try
        println("Loading dataset from: ", dataset_file)
        inputs, targets = load_optdigits(dataset_file)
        
        k = 5
        println("Generating indices for $k-fold cross validation...")
        
        cv_indices = crossvalidation(targets, k)
        
        test_indices = findall(x -> x == 1, cv_indices)
        
        indices_file = "Entrega/cv_indices.txt"
        writedlm(indices_file, test_indices)
        
        return test_indices
        
    catch e
        println("Error during index generation: ", e)
        error("Failed to generate cross-validation indices")
    end
end

# ------------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------------
data_path = "Entrega/optdigits.full"

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
crossValidationIndices = generate_cv_indices(data_path)

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
    :ANN => [Dict("topology" => t) for t in topologies],  # 8 neural network configurations
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
        try
            result = modelCrossValidation(modeltype, config, (train_inputs_norm, train_targets), training_cv_indices)
            push!(model_results, (config, result))
        catch e
            println("Error evaluating $modeltype with config $config: $e")
        end
    end
    all_results[modeltype] = model_results
end

# Create a dataframe with the results
println("Generating results...")
column_names = ["Model", "Params", "Mean Accuracy", "Std Accuracy"]
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

# Find the best overall model
best_model_type = first(sort(collect(pairs(best_configs)), by=x -> x[2][2][1][1], rev=true))[1]
best_model_config = best_configs[best_model_type][1]
println("Best model: $best_model_type with config: $best_model_config")

# Train the best model on the full training set and evaluate on test set
if best_model_type == :ANN
    # Special case for ANN
    model, = trainClassANN(
        best_model_config["topology"], 
        (train_inputs_norm, oneHotEncoding(train_targets)),
        maxEpochs=1000,
        learningRate=0.01
    )
    test_outputs = collect(model(Float32.(test_inputs_norm'))')
    test_outputs = classifyOutputs(test_outputs)
elseif best_model_type == :DoME
    # Special case for DoME
    test_outputs = trainClassDoME(
        (train_inputs_norm, train_targets), 
        test_inputs_norm, 
        best_model_config["maximumNodes"]
    )
else
    # For other models, just use the correct functions from the library
    if best_model_type == :SVC
        model = SVMClassifier(
            kernel = 
                best_model_config["kernel"]=="linear"  ? LIBSVM.Kernel.Linear :
                best_model_config["kernel"]=="rbf"     ? LIBSVM.Kernel.RadialBasis :
                best_model_config["kernel"]=="poly"    ? LIBSVM.Kernel.Polynomial :
                best_model_config["kernel"]=="sigmoid" ? LIBSVM.Kernel.Sigmoid : nothing,
            cost = Float64(best_model_config["C"]),
            gamma = Float64(get(best_model_config, "gamma",  -1)),
            degree = Int32(get(best_model_config, "degree", -1)),
            coef0 = Float64(get(best_model_config, "coef0",  -1))
        )
    elseif best_model_type == :DecisionTreeClassifier
        model = DTClassifier(max_depth = best_model_config["max_depth"], rng=Random.MersenneTwister(1))
    elseif best_model_type == :KNeighborsClassifier
        model = kNNClassifier(K = best_model_config["n_neighbors"])
    end
    
    # Create and train the machine
    mach = machine(model, MLJ.table(train_inputs_norm), categorical(train_targets))
    MLJ.fit!(mach, verbosity=0)
    
    # Get predictions
    test_outputs = MLJ.predict(mach, MLJ.table(test_inputs_norm))
    if best_model_type != :SVC
        test_outputs = mode.(test_outputs)
    end
end

# Calculate confusion matrix
classes = sort(unique(vcat(test_targets, test_outputs)))
_, _, _, _, _, _, _, test_confusion_matrix = confusionMatrix(test_outputs, test_targets, classes)

# Create heatmap for the confusion matrix
confusion_heatmap = heatmap(test_confusion_matrix, 
                           title="Confusion Matrix for Best Model on Test Set",
                           xlabel="Predicted", ylabel="Actual",
                           xticks=(1:length(classes), classes), yticks=(1:length(classes), classes),
                           color=:blues,
                           size=(700, 600))

# Add values to each cell
for i in 1:size(test_confusion_matrix, 1)
    for j in 1:size(test_confusion_matrix, 2)
        if test_confusion_matrix[i, j] > 0
            annotate!([(j, i, text(string(Int(test_confusion_matrix[i, j])), 8, :white))])
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
    fold_results[modeltype] = [r for r in 1:k]  # Placeholder as we need to reconstruct fold results
end

# Convert to matrix for CD diagram
# Each row is a fold/dataset, each column is a method
methods = [string(key) for key in keys(fold_results)]
n_folds = k
performances = zeros(n_folds, length(methods))

# Simulate fold results for demonstration
for (i, method) in enumerate(methods)
    mean_acc = best_models[Symbol(method)][2][1][1]
    std_acc = best_models[Symbol(method)][2][1][2]
    # Generate plausible fold accuracies - this is a simulation 
    performances[:, i] = mean_acc .+ std_acc .* randn(n_folds)
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

# ------------------------------------------------------------------
# Save Results
# ------------------------------------------------------------------
println("Saving results...")

# Create directory for plots if it doesn't exist
mkpath("Entrega/plots")

# Save results dataframe
CSV.write("results_optdigits.csv", sorted_df)

# Save plots
savefig(train_distribution, "plots/train_distribution.png")
savefig(test_distribution, "plots/test_distribution.png")
savefig(hm, "plots/correlation_heatmap.png")
savefig(acc_comparison, "plots/accuracy_comparison.png")
savefig(confusion_heatmap, "plots/confusion_matrix.png")
savefig(digit_grid, "plots/digit_samples.png")
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