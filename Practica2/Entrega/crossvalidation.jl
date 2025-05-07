
using Random
using Statistics
using Flux
using Plots
using DelimitedFiles
using Printf
using LIBSVM
using MLJ, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

include("73166321D_54157616E_48118254T_54152126Y.jl"); 

# Fijar la semilla aleatoria para reproducibilidad
Random.seed!(12345);

# Función para cargar índices desde un archivo
function load_cv_indices(filename::String)
    indices = vec(readdlm(filename, Int));
    return indices;
end;

# Función para cargar dataset 
function load_dataset(filename::String)
    data = readdlm(filename, ',');
    
    inputs = data[:, 1:64];
    inputs = Float32.(inputs)
    targets = data[:,65];
    return (inputs, targets);
end;

# Función para crear una tabla formateada con los resultados
function print_results_table(results, model_names)
    println("\n------------------------------------------------------------");
    println("                      RESULTADOS DE VALIDACIÓN CRUZADA");
    println("------------------------------------------------------------");
    println("Modelo         | Precisión      | Error         | F1-Score");
    println("------------------------------------------------------------");
    
    for (i, model_name) in enumerate(model_names)
        precision_mean, precision_std = results[i][1];
        error_mean, error_std = results[i][2];
        f1_mean, f1_std = results[i][7];
        
        @printf("%-14s | %.4f±%.4f | %.4f±%.4f | %.4f±%.4f\n", 
                model_name, precision_mean, precision_std, 
                error_mean, error_std, f1_mean, f1_std);
    end;
    println("------------------------------------------------------------");
end;

# Función para crear gráficas comparativas
function plot_comparison(results, model_names)
    metrics = ["Precisión", "Error", "Sensibilidad", "Especificidad", "VP+", "VP-", "F1-Score"];
    num_metrics = length(metrics);
    
    # Extraer medias y desviaciones estándar para cada métrica
    means = [results[i][j][1] for i in 1:length(model_names), j in 1:num_metrics];
    stds = [results[i][j][2] for i in 1:length(model_names), j in 1:num_metrics];
    
    # Crear gráficas
    p = plot(layout=(2,4), size=(1200,600), legend=:topright);
    
    for i in 1:num_metrics
        subplot = i <= 4 ? i : i+1;
        bar!(p[subplot], model_names, means[:,i], 
             yerror=stds[:,i], alpha=0.7, title=metrics[i],
             xlabel="Modelos", ylabel="Valor");
    end;
    
    # Añadir la matriz de confusión promedio del mejor modelo
    best_model_idx = argmax([r[1][1] for r in results]);
    heatmap!(p[5], results[best_model_idx][8], 
             title="Matriz de confusión\n($(model_names[best_model_idx]))", 
             xlabel="Predicción", ylabel="Real");
    
    return p;
end;

function main()
    # Cargar los índices de CV predefinidos
    indices_file = "Practica2/Entrega/cv_indices.txt";
    println("Cargando índices de validación cruzada desde: ", indices_file);
    
    try
        cv_indices = load_cv_indices(indices_file);
        println("Índices cargados: $(length(cv_indices)) valores");
        
        # Cargar el dataset
        dataset_file = "Practica2/Entrega/optical+recognition+of+handwritten+digits/optdigits.full";
        println("Cargando dataset desde: ", dataset_file);
        dataset = load_dataset(dataset_file);
        println("Dataset cargado: $(size(dataset[1], 1)) patrones con $(size(dataset[1], 2)) características");
        
        # Definir los modelos a evaluar
        models = [
            (:ANN, Dict("topology" => [5], "numExecutions" => 10, "maxEpochs" => 500, "learningRate" => 0.01)),
            (:DoME,Dict("maximumNodes" => [5])),
            (:SVC, Dict("kernel" => "rbf", "C" => 1.0, "gamma" => 0.1)),
            (:DecisionTreeClassifier, Dict("max_depth" => 5)),
            (:KNeighborsClassifier, Dict("n_neighbors" => 3))
        ];
        
        model_names = ["RNA [5]", "DoME[5]", "SVM RBF", "Árbol Decisión", "KNN-3"];
        
        # Ejecutar validación cruzada para cada modelo
        println("\nIniciando validación cruzada...");
        results = [];
        
        for (i, (model_type, hyperparams)) in enumerate(models)
            println("Evaluando modelo: $(model_names[i])");
            result = modelCrossValidation(model_type, hyperparams, dataset, cv_indices);
            push!(results, result);
            println("  ✓ Precisión: $(round(result[1][1], digits=4)) ± $(round(result[1][2], digits=4))");
        end;
        
        # Mostrar resultados en formato de tabla
        print_results_table(results, model_names);
        
        # Generar y guardar gráficas
        p = plot_comparison(results, model_names);
        savefig(p, "resultados_comparativa.png");
        println("\nGráfica de comparación guardada como: resultados_comparativa.png");
        
        # Mostrar la gráfica
        display(p);
        
    catch e
        println("Error durante la ejecución: ", e);
        return 1;
    end;
    
    return 0;
end;

exit(main());