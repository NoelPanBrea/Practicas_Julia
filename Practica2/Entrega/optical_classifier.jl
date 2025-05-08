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
include("index.jl")

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
# Paths para los archivos de training y testing
train_path = "Practica2/Entrega/optdigits.full"

# Función para cargar los datos
function load_optdigits(filename)
    data = readdlm(filename, ',')
    inputs = convert(Matrix{Float32}, data[:, 1:64])
    targets = string.(convert(Vector{Int64}, data[:, 65]))
    return inputs, targets
end

# Cargar los datos de entrenamiento y test
println("Cargando datos...")
train_inputs, train_targets = load_optdigits(train_path)

all_inputs = train_inputs
all_targets = train_targets

# Crear un dataframe para facilitar algunas visualizaciones
df_data = DataFrame(all_inputs, :auto)
df_data.target = all_targets

num_cols = size(df_data,2)
feature_cols = num_cols - 1

# Crear nombres de columnas para las 64 características
new_names = [Symbol("x$i") for i in 1:feature_cols]
    
# Renombrar solo las columnas de características, dejando la última (target) sin cambios
for i in 1:feature_cols
    rename!(df_data, names(df_data)[i] => new_names[i])
end

# Mostrar información básica
println("Dimensiones de los datos de entrenamiento: ", size(train_inputs))
println("Clases en el conjunto de entrenamiento: ", sort(unique(train_targets)))

# ------------------------------------------------------------------
# Plots y Análisis Exploratorio
# ------------------------------------------------------------------
println("Generando visualizaciones...")
colors = [:lightblue, :red, :green, :yellow, :orange, :purple, :cyan, :magenta, :lightgreen, :brown]

# Distribución de clases en el conjunto de entrenamiento
class_counts_train = countmap(train_targets)
class_labels_train = collect(keys(class_counts_train))
class_instances_train = collect(values(class_counts_train))
train_distribution = bar(class_labels_train, class_instances_train, color=colors,
                         legend=false, ylabel="Number of Instances",
                         xlabel="Digit", title="Digit Distribution in Training Set")

# Distribución de clases en el conjunto de test
class_counts_test = countmap(test_targets)
class_labels_test = collect(keys(class_counts_test))
class_instances_test = collect(values(class_counts_test))
test_distribution = bar(class_labels_test, class_instances_test, color=colors,
                       legend=false, ylabel="Number of Instances",
                       xlabel="Digit", title="Digit Distribution in Test Set")

# Histograma para algunos atributos seleccionados
selected_features = [1, 20, 40, 64]  # Seleccionamos algunos atributos para visualizar
histograms = [histogram(all_inputs[:, i], bins=20, title="Histogram of Feature $i", 
              label="Feature $i", legend=:outerright) for i in selected_features]

# Boxplot para algunos atributos seleccionados
boxplots = [boxplot(["Feature $i"], all_inputs[:, i], 
            title="Boxplot of Feature $i", label="Feature $i", 
            legend=:outerright) for i in selected_features]

# Matriz de correlación para una muestra de atributos
# (usar todos puede ser demasiado pesado visualmente)
sample_features = 1:10  # Primeros 10 atributos para la matriz de correlación
cor_matrix = cor(all_inputs[:, sample_features])
hm = heatmap(cor_matrix, title = "Feature Correlation Matrix (Sample)",
        xticks = (1:length(sample_features), ["Feature $i" for i in sample_features]),
        yticks = (1:length(sample_features), ["Feature $i" for i in sample_features]),
        color = :coolwarm,
        size=(800,800))

# Visualizar algunos dígitos como imágenes
function plot_digit(features, index)
    # Reshape de un vector 1x64 a una matriz 8x8
    digit = reshape(features[index, :], (8, 8))
    # Transponer para correcta visualización
    digit = permutedims(digit, (2, 1))
    # Visualizar
    heatmap(digit, color=:grays, aspect_ratio=:equal, 
           title="Digit: $(all_targets[index])", yflip=true, 
           xticks=false, yticks=false)
end

# Visualizar una muestra de dígitos
digit_indices = findall(x -> x in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], all_targets)
sample_indices = [findfirst(x -> x == d, all_targets) for d in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]]
digit_plots = [plot_digit(all_inputs, i) for i in sample_indices]
digit_grid = plot(digit_plots..., layout=(2, 5), size=(800, 400), legend=false)

# ------------------------------------------------------------------
# Normalización
# ------------------------------------------------------------------
println("Normalizando datos...")
# Calculamos parámetros de normalización usando solo el conjunto de entrenamiento
min_vals, max_vals = calculateMinMaxNormalizationParameters(train_inputs)

# Normalizamos ambos conjuntos con los mismos parámetros
train_inputs_norm = normalizeMinMax(train_inputs, (min_vals, max_vals))

# ------------------------------------------------------------------
# Model Setup
# ------------------------------------------------------------------
println("Preparando configuración de modelos...")

# Creamos índices de cross-validation estratificados basados en las clases
crossValidationIndices = vec(readdlm("Practica2/Entrega/cv_indices.txt", Int64))

# Definimos las configuraciones para cada modelo
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
    :ANN => [Dict("topology" => t) for t in topologies],  # 8 configuraciones de redes neuronales
    :SVC => [Dict("kernel" => k, "C" => c, "gamma" => 0.1, "coef0" => 0.5, "degree" => 3) 
             for k in ["linear", "rbf", "poly", "sigmoid"] for c in [1, 10]],  # 8 configuraciones SVM
    :DoME => [Dict("maximumNodes" => n) for n in 5:12], # 8 valores de nodos
    :DecisionTreeClassifier => [Dict("max_depth" => d) for d in 7:12], # 6 valores de profundidad
    :KNeighborsClassifier => [Dict("n_neighbors" => k) for k in [1,3,5,7,9,11]]  # 6 valores de vecinos
)

# ------------------------------------------------------------------
# Model Training and Evaluation
# ------------------------------------------------------------------
println("Entrenando y evaluando modelos...")
all_results = Dict()
model_configuration_array = collect(pairs(model_configurations))

# Ejecutamos todos los modelos con sus configuraciones
for (modeltype, configs) in model_configuration_array
    println("Evaluando modelo: $modeltype")
    model_results = []
    for config in configs
        println("- Configuración: $config")
        result = modelCrossValidation(modeltype, config, (train_inputs_norm, train_targets), crossValidationIndices)
        push!(model_results, (config, result))
    end
    all_results[modeltype] = model_results
end

# Mostrar los resultados en un dataframe
println("Generando resultados...")
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

# Ordenar por Mean Accuracy
sorted_df = sort(df_result, :Mean_Accuracy, rev=true)

# ------------------------------------------------------------------
# Evaluación del modelo en conjunto de test
# ------------------------------------------------------------------
println("Evaluando mejor modelo en conjunto de test...")

# Encontrar el mejor modelo
best_model_row = sorted_df[1, :]
best_model_type = Symbol(best_model_row.Model)
best_model_config = best_model_row.Params

println("Mejor modelo: $best_model_type con configuración $best_model_config")

# Evaluación especial para el conjunto de test
# Necesitaríamos implementar una función específica según el tipo de modelo

# ------------------------------------------------------------------
# Accuracy Comparation
# ------------------------------------------------------------------
println("Comparando precisión de modelos...")
best_configs = Dict()
for (modeltype, results) in all_results
    # Ordenamos los resultados para cada modelo y tomamos el mejor
    best_result = sort(results, by=x -> x[2][1][1], rev=true)[1]
    best_configs[modeltype] = best_result
end

# Obtenemos los modelos del Dict para poder graficar
model_types = [string(k) for k in keys(best_configs)]
# Extraemos las accuracies
accuracies = [v[2][1][1] for v in values(best_configs)]

# Ajustamos las gráficas para mejorar su visualización
min_accuracy, max_accuracy = minimum(accuracies), maximum(accuracies)
padding = (max_accuracy - min_accuracy) * 0.1
ylims_range = (min_accuracy - padding, max_accuracy + padding * 2)

acc_comparison = bar(model_types, accuracies, legend=false,
    ylabel="Accuracy (%)", xlabel="Model Type",
    title="Comparison of Best Model Accuracies",
    ylims=ylims_range,
    yticks=round(min_accuracy - padding, digits=3):0.005:round(max_accuracy + padding * 2, digits=3),
    bar_width=0.5,
    color=[:lightblue, :lightgreen, :lightcoral, :lightpink],
    size=(800, 600))

# Para cada barra añadimos el porcentaje    
annotate!([(i, accuracies[i] + 0.002, text(string(round(accuracies[i] * 100, digits=2)) * "%", 10)) for i in 1:length(accuracies)])

# ------------------------------------------------------------------
# Confusion Matrix para el mejor modelo
# ------------------------------------------------------------------
println("Generando matriz de confusión para el mejor modelo...")
# Extraemos la matriz de confusión del mejor resultado global
best_overall_model_type = Symbol(sorted_df[1, :Model])
best_overall_model_config = sorted_df[1, :Params]
best_overall_results = nothing

for (modeltype, results) in all_results
    if modeltype == best_overall_model_type
        for result in results
            if result[1] == best_overall_model_config
                best_overall_results = result[2]
                break
            end
        end
    end
end

# Extraer la matriz de confusión
confusion_matrix = best_overall_results[8]

# Crear heatmap de la matriz de confusión
confusion_heatmap = heatmap(confusion_matrix, 
                           title="Confusion Matrix for Best Model",
                           xlabel="Predicted", ylabel="Actual",
                           xticks=(1:10, 0:9), yticks=(1:10, 0:9),
                           color=:blues,
                           size=(700, 600))

# Añadir valores a cada celda
for i in 1:size(confusion_matrix, 1)
    for j in 1:size(confusion_matrix, 2)
        if confusion_matrix[i, j] > 0
            annotate!([(j, i, text(string(confusion_matrix[i, j]), 8, :white))])
        end
    end
end

# ------------------------------------------------------------------
# Diagrama de Diferencia Crítica (CD)
# ------------------------------------------------------------------
println("Generando diagrama de diferencia crítica...")

# Extraer los resultados de todos los modelos para cada fold
# Necesitamos una matriz donde cada fila sea un dataset/fold y cada columna un método
fold_results = Dict()

# Crear un diccionario para almacenar los mejores resultados por tipo de modelo
best_models = Dict()
for (modeltype, results) in all_results
    # Encontrar la mejor configuración para cada tipo de modelo
    best_config_index = argmax([r[2][1][1] for r in results])
    best_models[modeltype] = results[best_config_index]
    
    # Extraer resultados por fold para la mejor configuración
    # En los resultados, [2][5] es el vector con la accuracy de cada fold
    fold_results[modeltype] = best_models[modeltype][2][5]
end

# Convertir a matriz para el diagrama CD
# Cada fila es un fold/dataset, cada columna es un método
methods = [string(key) for key in keys(fold_results)]
n_folds = length(first(values(fold_results)))
performances = zeros(n_folds, length(methods))

for (i, method) in enumerate(methods)
    performances[:, i] = fold_results[Symbol(method)]
end

# Crear el diagrama CD
cd_diagram = create_cd_diagram(
    methods, 
    performances, 
    α=0.05, 
    lower_is_better=false,  # Usamos accuracy, por lo que valores más altos son mejores, si no true
    title="Comparación de Modelos ML",
    figsize=(900, 500)
)


display(cd_diagram)

# ------------------------------------------------------------------
# Save the Results
# ------------------------------------------------------------------
println("Guardando resultados...")

# Crear directorio para plots si no existe
mkpath("plots")

# Guardar dataframe de resultados
CSV.write("results_optdigits.csv", sorted_df)

# Guardar plots
savefig(train_distribution, "plots/train_distribution.png")
savefig(test_distribution, "plots/test_distribution.png")
savefig(hm, "plots/correlation_heatmap.png")
savefig(acc_comparison, "plots/accuracy_comparison.png")
savefig(confusion_heatmap, "plots/confusion_matrix.png")
savefig(digit_grid, "plots/digit_samples.png")

# Guardar diagrama
savefig(cd_diagram, "plots/cd_diagram.png")

# Guardar histogramas seleccionados
for (i, idx) in enumerate(selected_features)
    savefig(histograms[i], "plots/histogram_feature$(idx).png")
end

# Guardar boxplots seleccionados
for (i, idx) in enumerate(selected_features)
    savefig(boxplots[i], "plots/boxplot_feature$(idx).png")
end

println("Análisis completado y resultados guardados.")