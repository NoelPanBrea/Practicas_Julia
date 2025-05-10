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

# ------------------------------------------------------------------
# Critical Difference Diagram Implementation
# ------------------------------------------------------------------
using Plots
using Statistics
using StatsBase
using Random

function create_cd_diagram(methods, performances; α=0.05, lower_is_better=true, title="", figsize=(800, 400))
    n_methods = length(methods)
    n_datasets = size(performances, 1)
    q_alpha_values = [0, 1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]
    # Calcular rangos para cada dataset
    ranks = zeros(n_datasets, n_methods)
    for i in 1:n_datasets
        if lower_is_better
            # Para métricas donde valores más bajos son mejores (como error)
            ranks[i, :] = tiedrank(performances[i, :])
        else
            # Para métricas donde valores más altos son mejores (como exactitud)
            ranks[i, :] = tiedrank(1.0 .- performances[i, :])
        end
    end
    
    # Calcular rangos medios
    avg_ranks = vec(mean(ranks, dims=1))
    
    # Calcular diferencia crítica según el test de Nemenyi
    # q_alpha = quantile(StudentizedRange(n_methods), 1 - α)
    q_alpha = q_alpha_values[n_methods]
    cd = q_alpha * sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    # Ordenar métodos por rango medio
    sorted_indices = sortperm(avg_ranks)
    sorted_methods = methods[sorted_indices]
    sorted_ranks = avg_ranks[sorted_indices]
    
    # Crear el gráfico
    plt = plot(
        xlim=(0.5, n_methods + 0.5),
        ylim=(0, n_methods + 1),
        xticks=1:n_methods,
        yticks=[],
        grid=false,
        legend=false,
        size=figsize,
        margin=10Plots.mm,
        title=title
    )
    
    # Dibujar línea horizontal superior para la escala
    plot!(plt, [1, n_methods], [n_methods + 0.5, n_methods + 0.5], color=:black, linewidth=1)
    
    # Dibujar líneas verticales para la escala
    for i in 1:n_methods
        plot!(plt, [i, i], [n_methods + 0.3, n_methods + 0.7], color=:black, linewidth=1)
    end
    
    # Dibujar CD en la parte superior
    cd_start = 1.325
    cd_end = cd_start + cd
    plot!(plt, [cd_start, cd_end], [n_methods + 1, n_methods + 1], color=:black, linewidth=2)
    annotate!(plt, [(cd_start + cd/2, n_methods + 1.2, "CD = $(round(cd, digits=2))")])
    
    # Dibujar líneas horizontales para cada algoritmo y añadir nombres
    for (i, (method, rank)) in enumerate(zip(sorted_methods, sorted_ranks))
        y_pos = n_methods - i + 1
        # Línea desde el eje y hasta el rango
        plot!(plt, [1, rank], [y_pos, y_pos], color=:black, linewidth=1)
        # Línea desde el eje x hasta el rango
        plot!(plt, [rank, rank], [n_methods + 0.5, y_pos], color=:black, linewidth=1)
        # Añadir nombre del método
        annotate!(plt, [(0.97, y_pos, (method, :right, 9))])
    end
    
    # Identificar y dibujar grupos de métodos no significativamente diferentes
    groups = []
    current_group = [1]
    
    for i in 2:length(sorted_ranks)
        if sorted_ranks[i] - sorted_ranks[current_group[1]] <= cd
            push!(current_group, i)
        else
            push!(groups, copy(current_group))
            current_group = [i]
        end
    end
    push!(groups, current_group)
    
    # Dibujar líneas horizontales para grupos
    for (group_idx, group) in enumerate(groups)
        if length(group) > 1
            min_rank = sorted_ranks[group[1]]
            max_rank = sorted_ranks[group[end]]
            y_pos = n_methods - group[1] + 1 + 0.2
            plot!(plt, [min_rank - 0.1, max_rank + 0.1], [y_pos, y_pos], color=:black, linewidth=2)
        end
    end
    
    return plt
end

# ------------------------------------------------------------------
# Load Function
# ------------------------------------------------------------------

function load_optdigits(filename)
    data = readdlm(filename, ',')
    inputs = convert(Matrix{Float32}, data[:, 1:64])
    targets = string.(convert(Vector{Int64}, data[:, 65]))
    return inputs, targets
end

# ------------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------------

# Load all data
println("Loading data...")
data_path  = "Entrega/optdigits.full"
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

# number of folds
k = 5

training_cv_indices = crossvalidation(all_targets, k)

# Create train and test sets

# Display basic information
clases = sort(unique(all_targets))
println("Dataset dimensions: ", size(all_inputs))
println("Numer of classes in dataset: ", clases)

# ------------------------------------------------------------------
# Exploratory Data Analysis
# ------------------------------------------------------------------
println("Generating visualizations...")
colors = [:lightblue, :red, :green, :yellow, :orange, :purple, :cyan, :magenta, :lightgreen, :brown]

# Class distribution in the training set
class_counts_train = countmap(all_targets)
class_labels_train = collect(keys(class_counts_train))
class_instances_train = collect(values(class_counts_train))
class_distribution = bar(class_labels_train, class_instances_train, ylims = (500, 600), color=colors,
                         legend=false, ylabel="Number of Instances",
                         xlabel="Digit", title="Digit Distribution in the dataset")

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
min_vals, max_vals = calculateMinMaxNormalizationParameters(all_inputs)

# Normalize both sets with the same parameters
all_inputs_norm = normalizeMinMax(all_inputs, (min_vals, max_vals))

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
    :ANN => [Dict("topology" => t, "maxEpochs" => 100) for t in topologies],  # 8 neural network configurations
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

# Run cross-validation for each model configuration
for (modeltype, configs) in model_configuration_array
    println("Evaluating model: $modeltype")
    model_results = []
    for config in configs
        println("- Configuration: $config")
        try
            result = modelCrossValidation(modeltype, config, (all_inputs_norm, all_targets), training_cv_indices)
            push!(model_results, (config, result))
        catch e
            println("Error evaluating $modeltype with config $config: $e")
        end
    end
    all_results[modeltype] = model_results
end

# Create a dataframe with the results
println("Generating results...")
column_names = ["Model", "Params",
"Mean Accuracy", "Std Accuracy",
"Mean Error Rate", "Std Error Rate",
"Mean Sensitivity", "Std Sensitivity",
"Mean Specificity", "Std Specificity",
"Mean Precision", "Std Precision",
"Mean NPV", "Std NPV",
"Mean F1", "Std F1"]
df_result = DataFrame(Model=String[], Params=Any[],
Mean_Accuracy=Float64[], Std_Accuracy=Float64[],
Mean_Error_Rate=Float64[], Std_Error_Rate=Float64[],
Mean_Sensitivity=Float64[], Std_Sensitivity=Float64[],
Mean_Specificity=Float64[], Std_Specificity=Float64[],
Mean_Precision=Float64[], Std_Precision=Float64[],
Mean_NPV=Float64[], Std_NPV=Float64[],
Mean_F1=Float64[], Std_F1=Float64[],

    )
for (modeltype, results) in all_results
    for result in results
        config = result[1]
        mean_acc = result[2][1][1]
        std_acc = result[2][1][2]
        push!(df_result, (String(modeltype), config,
        mean_acc, std_acc,
        result[2][2][1], result[2][2][2],
        result[2][3][1], result[2][3][2],
        result[2][4][1], result[2][4][2],
        result[2][5][1], result[2][5][2],
        result[2][6][1], result[2][6][2],
        result[2][7][1], result[2][7][2]))
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

# ------------------------------------------------------------------
# Confusion Matrix for the Best Model of each kind
# ------------------------------------------------------------------
println("Plotting confusion matrix for the best model of each kind...")

# Find the best overall model
confusion_heatmaps = []
for (modelType, result) in collect(pairs(best_configs))
    # Create heatmap for the confusion matrix
    println("Plotting confusion matrix for : $modelType $(result[1]) ")
    confusion_heatmap = heatmap(result[2][8], 
                            title="Confusion Matrix for $modelType",
                            xlabel="Predicted", ylabel="Actual",
                            xticks=(1:length(clases), clases), yticks=(1:length(clases), clases),
                            color=:blues,
                            size=(700, 600))

    # Add values to each cell
    for i in 1:size(result[2][8], 1)
        for j in 1:size(result[2][8], 2)
            if result[2][8][i, j] > 0
                annotate!([(j, i, text(string(Int(round(result[2][8][i, j]))), 8, :white))])
            end
        end
    end
    push!(confusion_heatmaps, (modelType, confusion_heatmap))
end


# ------------------------------------------------------------------
# Critical Difference Diagram
# ------------------------------------------------------------------
println("Generating critical difference diagram...")


performances = zeros(k, length(best_configs))
for (j, (modelType, result)) in enumerate(collect(pairs(best_configs)))
    fold_performance = []
    for i in 1:k
        train_inputs_fold = all_inputs_norm[findall(x -> (x != i), training_cv_indices), :];
        train_targets_fold = all_targets[findall(x -> (x != i), training_cv_indices)];
        test_inputs_fold = all_inputs_norm[findall(x -> (x == i), training_cv_indices), :];
        test_targets_fold = all_targets[findall(x -> (x == i), training_cv_indices)];
        if modelType == :ANN
            # Special case for ANN
            model, = trainClassANN(
                result[1]["topology"], 
                (train_inputs_fold, oneHotEncoding(train_targets_fold)),
                maxEpochs=1000,
                learningRate=0.01
            )
            test_outputs = accuracy(model(Float32.(test_inputs_fold'))', oneHotEncoding(test_targets_fold))
        elseif modelType == :DoME
            # Special case for DoME
            test_outputs = trainClassDoME(
                (train_inputs_fold, train_targets_fold), 
                test_inputs_fold, 
                result[1]["maximumNodes"]
            )
        else
            if modelType == :SVC  # For other models, just use the correct functions from the library

                model = SVMClassifier(
                    kernel = 
                        result[1]["kernel"]=="linear"  ? LIBSVM.Kernel.Linear :
                        result[1]["kernel"]=="rbf"     ? LIBSVM.Kernel.RadialBasis :
                        result[1]["kernel"]=="poly"    ? LIBSVM.Kernel.Polynomial :
                        result[1]["kernel"]=="sigmoid" ? LIBSVM.Kernel.Sigmoid : nothing,
                    cost = Float64(result[1]["C"]),
                    gamma = Float64(get(result[1], "gamma",  -1)),
                    degree = Int32(get(result[1], "degree", -1)),
                    coef0 = Float64(get(result[1], "coef0",  -1))
                )
            elseif modelType == :DecisionTreeClassifier
                model = DTClassifier(max_depth = result[1]["max_depth"], rng=Random.MersenneTwister(1))
            elseif modelType == :KNeighborsClassifier
                model = kNNClassifier(K = result[1]["n_neighbors"])
            end
        
        
            # Create and train the machine
            mach = machine(model, MLJ.table(train_inputs_fold), categorical(train_targets_fold))
            MLJ.fit!(mach, verbosity=0)
            
            # Get predictions
            test_outputs = MLJ.predict(mach, MLJ.table(test_inputs_fold))
            if modelType != :SVC
                test_outputs = mode.(test_outputs)
            end
            test_outputs = accuracy(oneHotEncoding(test_outputs), oneHotEncoding(test_targets_fold))
        end
        if modelType == :DoME
            test_outputs = accuracy(oneHotEncoding(test_outputs), oneHotEncoding(test_targets_fold))
        end
        push!(fold_performance, test_outputs)
    end
    performances[:, j] = fold_performance
end

# Create CD diagram
method_list = ["DTC", "SVC", "DoME", "KNN", "ANN"]
cd_diagram = create_cd_diagram(
    Vector{String}(method_list), 
    performances, 
    α=0.05, 
    lower_is_better=false,  # We use accuracy, so higher values are better
    title="ML Model Comparison",
    figsize=(900, 500)
)

# ------------------------------------------------------------------
# Generación de tablas detalladas con métricas
# ------------------------------------------------------------------
println("Generando tablas de métricas para cada modelo...")

# Directorio para guardar las tablas
tables_dir = "Entrega/Tablas"
mkpath(tables_dir)

# Función para generar tabla LaTeX con resultados detallados
function generate_normal_table(model_type::Symbol, results, column_header::String)
    filename = joinpath(tables_dir, "tabla_$(string(model_type)).txt")
    
    open(filename, "w") do file
        # Crear un título según el tipo de modelo
        title = if model_type == :ANN
            "Rendimiento de diferentes topologías de redes neuronales artificiales (ANN)"
        elseif model_type == :SVC
            "Rendimiento de diferentes configuraciones de SVM"
        elseif model_type == :DoME
            "Rendimiento de diferentes configuraciones de DoME"
        elseif model_type == :DecisionTreeClassifier
            "Rendimiento de diferentes configuraciones de árboles de decisión"
        elseif model_type == :KNeighborsClassifier
            "Rendimiento de diferentes configuraciones de k-NN"
        end
        
        # Escribir el título
        write(file, title * "\n")
        write(file, "=" ^ length(title) * "\n\n")
        
        # Crear encabezados con formato fijo
        header = string(
            rpad(column_header, 15), " | ",
            rpad("Precisión", 10), " | ",
            rpad("Desv. Est.", 10), " | ",
            rpad("Error", 10), " | ",
            rpad("Sensibilidad", 10), " | ",
            rpad("Especificidad", 10), " | ",
            rpad("F1-Score", 10)
        )
        write(file, header * "\n")
        
        # Línea divisoria
        divider = "-" ^ 90
        write(file, divider * "\n")
        
        # Encontrar el índice del mejor resultado
        best_index = argmax([r[2][1][1] for r in results])
        
        # Añadir filas para cada configuración
        for (i, result) in enumerate(results)
            config = result[1]
            metrics = result[2]
            
            # Extraer las métricas
            mean_acc, std_acc = metrics[1]
            mean_err, _ = metrics[2]
            mean_sens, _ = metrics[3]
            mean_spec, _ = metrics[4]
            mean_f1, _ = metrics[7]
            
            # Formatear la configuración como string según el tipo de modelo
            config_str = if model_type == :ANN
                "[" * join(string.(config["topology"]), ",") * "]"
            elseif model_type == :SVC
                config["kernel"] * "-C" * string(config["C"])
            elseif model_type == :DoME
                string(config["maximumNodes"])
            elseif model_type == :DecisionTreeClassifier
                string(config["max_depth"])
            elseif model_type == :KNeighborsClassifier
                string(config["n_neighbors"])
            else
                string(config)
            end
            
            # Marcar con * el mejor resultado
            if i == best_index
                config_str = config_str * " *"
            end
            
            # Crear la fila con formato fijo
            row = string(
                rpad(config_str, 15), " | ",
                rpad(string(round(mean_acc, digits=4)), 10), " | ",
                rpad(string(round(std_acc, digits=4)), 10), " | ",
                rpad(string(round(mean_err, digits=4)), 10), " | ",
                rpad(string(round(mean_sens, digits=4)), 10), " | ",
                rpad(string(round(mean_spec, digits=4)), 10), " | ",
                rpad(string(round(mean_f1, digits=4)), 10)
            )
            write(file, row * "\n")
        end
        
        write(file, divider * "\n")
        write(file, "* = Mejor configuración\n")
    end
    
    # También mostrar la tabla en la consola
    println("\n=== TABLA DE RESULTADOS PARA $model_type ===")
    
    # Encabezados para consola
    header = string(
        rpad(column_header, 15), " | ",
        rpad("Precisión", 10), " | ",
        rpad("Desv. Est.", 10), " | ",
        rpad("Error", 10), " | ",
        rpad("Sensib.", 10), " | ",
        rpad("Especif.", 10), " | ",
        rpad("F1-Score", 10)
    )
    println(header)
    
    # Línea divisoria
    divider = "-" ^ 90
    println(divider)
    
    # Imprimir filas para consola
    best_index = argmax([r[2][1][1] for r in results])
    for (i, result) in enumerate(results)
        config = result[1]
        metrics = result[2]
        
        # Extraer las métricas
        mean_acc, std_acc = metrics[1]
        mean_err, _ = metrics[2]
        mean_sens, _ = metrics[3]
        mean_spec, _ = metrics[4]
        mean_f1, _ = metrics[7]
        
        # Formatear la configuración como string según el tipo de modelo
        config_str = if model_type == :ANN
            "[" * join(string.(config["topology"]), ",") * "]"
        elseif model_type == :SVC
            config["kernel"] * "-C" * string(config["C"])
        elseif model_type == :DoME
            string(config["maximumNodes"])
        elseif model_type == :DecisionTreeClassifier
            string(config["max_depth"])
        elseif model_type == :KNeighborsClassifier
            string(config["n_neighbors"])
        else
            string(config)
        end
        
        # Marcar con * el mejor resultado
        if i == best_index
            config_str = config_str * " *"
        end
        
        # Crear la fila con formato fijo para consola
        row = string(
            rpad(config_str, 15), " | ",
            rpad(string(round(mean_acc, digits=4)), 10), " | ",
            rpad(string(round(std_acc, digits=4)), 10), " | ",
            rpad(string(round(mean_err, digits=4)), 10), " | ",
            rpad(string(round(mean_sens, digits=4)), 10), " | ",
            rpad(string(round(mean_spec, digits=4)), 10), " | ",
            rpad(string(round(mean_f1, digits=4)), 10)
        )
        println(row)
    end
    
    println(divider)
    println("* = Mejor configuración")
    println()
    
    println("Tabla para $model_type generada: $filename")
    return nothing
end

# Generar tablas para cada tipo de modelo
for (modeltype, results) in all_results
    column_header = if modeltype == :ANN
        "Topología"
    elseif modeltype == :SVC
        "Configuración"
    elseif modeltype == :DoME
        "Nodos Máx."
    elseif modeltype == :DecisionTreeClassifier
        "Prof. Máx."
    elseif modeltype == :KNeighborsClassifier
        "Vecinos"
    else
        "Configuración"
    end
    
    generate_normal_table(modeltype, results, column_header)
end
# ------------------------------------------------------------------
# Save Results
# ------------------------------------------------------------------
println("Saving results...")

# Create directory for plots if it doesn't exist
mkpath("Entrega/plots")
mkpath("Entrega/plots/ConfusionHeatmaps")


# Save results dataframe
CSV.write("Entrega/results_optdigits.csv", df_result)

# Save plots
savefig(class_distribution, "Entrega/plots/class_distribution")
savefig(hm, "Entrega/plots/correlation_heatmap")
# savefig(acc_comparison, "Entrega/plots/accuracy_comparison")
for (modelType, confusion_heatmap) in confusion_heatmaps
    savefig(confusion_heatmap, "Entrega/plots/ConfusionHeatmaps/$(modelType)confusion_matrix")
end
savefig(digit_grid, "Entrega/plots/digit_samples")
savefig(cd_diagram, "Entrega/plots/cd_diagram")

println("Analysis completed and results saved.")
