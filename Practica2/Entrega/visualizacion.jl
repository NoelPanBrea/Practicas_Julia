using Random
using Statistics
using Flux
using Plots
using DelimitedFiles
using Printf
using StatsPlots
using DataFrames

include("73166321D_54157616E_48118254T_54152126Y.jl");

# Fijar la semilla aleatoria para reproducibilidad
Random.seed!(12345);

# Función para cargar índices desde un archivo
function load_cv_indices(filename::String)
    indices = vec(readdlm(filename, Int));
    return indices;
end;

# Función para cargar dataset (ajustar según el formato de tus datos)
function load_dataset(filename::String)
    data = readdlm(filename, ',');
    # Asumiendo que la última columna es la etiqueta
    inputs = convert(Array{Float64,2}, data[:, 1:end-2]);
    targets = convert(Array{Any,1}, data[:, end-1]);
    return (inputs, targets);
end;

# Función para imprimir los resultados detallados de un modelo
function print_model_details(model_name, result)
    println("\n----------------------------------------------------------");
    println("RESULTADOS DETALLADOS PARA: $model_name");
    println("----------------------------------------------------------");
    
    metrics = ["Precisión", "Error", "Sensibilidad", "Especificidad", 
               "Valor Predictivo Positivo", "Valor Predictivo Negativo", "F1-Score"];
    
    for (i, metric) in enumerate(metrics)
        mean_val, std_val = result[i];
        @printf("%s: %.4f ± %.4f\n", metric, mean_val, std_val);
    end
    
    println("\nMatriz de confusión:");
    println(result[8]);
    println("----------------------------------------------------------");
end;

# Función para crear un gráfico de barras comparativo para todas las métricas
function plot_metrics_comparison(results, model_names)
    metrics = ["Precisión", "Error", "Sensibilidad", "Especificidad", 
               "Valor Predictivo Positivo", "Valor Predictivo Negativo", "F1-Score"];
    
    # Preparar datos para el gráfico
    df = DataFrame();
    df.Modelo = repeat(model_names, inner=7);
    df.Métrica = repeat(metrics, outer=length(model_names));
    df.Valor = [results[i÷7+1][i%7+1][1] for i in 0:(length(model_names)*7-1)];
    df.Std = [results[i÷7+1][i%7+1][2] for i in 0:(length(model_names)*7-1)];
    
    # Crear el gráfico
    p = groupedbar(df.Modelo, df.Valor, group=df.Métrica,
                  yerror=df.Std, 
                  title="Comparación de métricas por modelo",
                  xlabel="Modelo", ylabel="Valor",
                  legend=:outertopright,
                  size=(900, 600),
                  bar_width=0.7,
                  lw=0,
                  framestyle=:box);
    
    return p;
end;

# Función para crear un heatmap de matrices de confusión
function plot_confusion_matrices(results, model_names)
    n_models = length(model_names);
    plots = [];
    
    for i in 1:n_models
        cm = results[i][8];
        # Normalizar la matriz por filas (reales)
        cm_norm = cm ./ sum(cm, dims=2);
        
        p = heatmap(cm_norm, 
                   title=model_names[i],
                   xlabel="Predicción", ylabel="Real",
                   color=:viridis,
                   aspect_ratio=1,
                   clim=(0,1),
                   annotations=[(j, i, text(round(cm_norm[i,j], digits=2), 8, :white))
                               for i in 1:size(cm_norm,1) for j in 1:size(cm_norm,2)]);
        push!(plots, p);
    end;
    
    return plot(plots..., layout=(2, ceil(Int, n_models/2)), size=(800, 800));
end;

# Función para crear una curva de aprendizaje para una RNA
function plot_learning_curve(topology, dataset, cv_indices; fold=1)
    # Extraer los datos para un fold específico
    fold_train_indices = findall(x -> x != fold, cv_indices)
    fold_test_indices = findall(x -> x == fold, cv_indices)
    
    fold_train_inputs = dataset[1][fold_train_indices, :]
    fold_train_targets = dataset[2][fold_train_indices]
    fold_test_inputs = dataset[1][fold_test_indices, :]
    fold_test_targets = dataset[2][fold_test_indices]
    
    # Convertir las etiquetas a matriz one-hot
    classes = unique(dataset[2])
    fold_train_targets_oh = oneHotEncoding(fold_train_targets, classes)
    fold_test_targets_oh = oneHotEncoding(fold_test_targets, classes)
    
    # Entrenar el modelo con registro de pérdidas
    ann, train_losses, val_losses, test_losses = trainClassANN(
        topology, 
        (fold_train_inputs, fold_train_targets_oh),
        testDataset=(fold_test_inputs, fold_test_targets_oh),
        maxEpochs=300,
        learningRate=0.01
    )
    
    # Crear la gráfica
    p = plot(train_losses, 
             label="Entrenamiento", 
             title="Curva de aprendizaje - RNA $topology",
             xlabel="Época", 
             ylabel="Pérdida",
             lw=2);
    
    plot!(p, test_losses, 
          label="Test", 
          lw=2);
    
    return p, ann;
end;
function compare_in_between()
    dataset = load_dataset("Practica2/Entrega/optical+recognition+of+handwritten+digits/optdigits.full")
    cv_indices = load_cv_indices("Practica2/Entrega/cv_indices.txt")
    display(compare_numerical_models(dataset, cv_indices, :KNeighborsClassifier, 20)[1])
end

function compare_svc_configs(dataset, cv_indices)
   results = []
end

function compare_numerical_models(dataset, cv_indices, modelType, n)
    range = 1:3
    results = []
    names = []
    hyperparameter_type = (modelType == :KNeighborsClassifier) ? "n_neighbors" : "max_depth"
    for i in range
        println("Evaluando Modelo $modelType con parámeto $hyperparameter_type : $i")
        result = modelCrossValidation(modelType, Dict(hyperparameter_type => i), dataset, cv_indices)
        name = "$hyperparameter_type $i"
        push!(results, result)
        push!(names, name)
    end
    # Graficar comparación
    accuracy_means = [r[1][1] for r in results]
    accuracy_stds = [r[1][2] for r in results]
    
    p = bar(names, accuracy_means,
            yerror=accuracy_stds,
            title="Comparación de configuraciones $modelType",
            xlabel="configuraciones", 
            ylabel="precision",
            legend=false,
            color=:skyblue)
    return p, results, names
end

# Función para comparar diferentes topologías de RNA
function compare_ann_topologies(dataset, cv_indices)
    topologies = [[3], [5], [10], [5, 3], [10, 5]]
    topology_names = ["[3]", "[5]", "[10]", "[5, 3]", "[10, 5]"]
    results = []
    
    # Evaluar cada topología
    for (i, topology) in enumerate(topologies)
        println("Evaluando RNA con topología: ", topology_names[i])
        result = ANNCrossValidation(
            topology, 
            dataset, 
            cv_indices,
            numExecutions=5,
            maxEpochs=300,
            learningRate=0.01
        )
        push!(results, result)
    end
    
    # Graficar comparación
    accuracy_means = [r[1][1] for r in results]
    accuracy_stds = [r[1][2] for r in results]
    
    p = bar(topology_names, accuracy_means,
            yerror=accuracy_stds,
            title="Comparación de topologías RNA",
            xlabel="Topología", 
            ylabel="Exactitud",
            legend=false,
            color=:skyblue)
    
    return p, results, topology_names
end

# Principal: Ejecutar experimentos completos
function main()
    # Cargar los índices de CV predefinidos
    cv_indices_file = "cv_indices.txt";  
    println("Cargando índices de validación cruzada desde: ", cv_indices_file);
    
    try
        cv_indices = load_cv_indices(cv_indices_file);
        println("Índices cargados: $(length(cv_indices)) valores");
        
        # Cargar el dataset
        dataset_file = "Practica2/Entrega/optical+recognition+of+handwritten+digits/optdigits.full";
        println("Cargando dataset desde: ", dataset_file);
        dataset = load_dataset(dataset_file);
        println("Dataset cargado: $(size(dataset[1], 1)) patrones con $(size(dataset[1], 2)) características");
        
        # Definir los modelos a evaluar
        models = [
            (:ANN, Dict("topology" => [5], "numExecutions" => 5, "maxEpochs" => 300, "learningRate" => 0.01)),
            (:DoME, Dict("maximumNodes" => [5])),
            (:SVC, Dict("kernel" => "rbf", "C" => 1.0, "gamma" => 0.1)),
            (:DecisionTreeClassifier, Dict("max_depth" => 5)),
            (:KNeighborsClassifier, Dict("n_neighbors" => 3)),
            (:DoME, Dict("maximumNodes" => 20))
        ]
        
        model_names = ["RNA", "DoME","SVM", "Árbol Decisión", "KNN", "DoME"]
        
        # Ejecutar validación cruzada para cada modelo
        println("\nIniciando validación cruzada...");
        results = [];
        
        for (i, (model_type, hyperparams)) in enumerate(models)
            println("Evaluando modelo: $(model_names[i])", hyperparams);
            result = modelCrossValidation(model_type, hyperparams, dataset, cv_indices);
            push!(results, result);
            print_model_details(model_names[i], result);
        end;
        
        # Generar gráficas comparativas
        p1 = plot_metrics_comparison(results, model_names);
        savefig(p1, "comparacion_metricas.png");
        println("\nGráfica de comparación de métricas guardada como: comparacion_metricas.png");
        
        p2 = plot_confusion_matrices(results, model_names);
        savefig(p2, "matrices_confusion.png");
        println("Gráfica de matrices de confusión guardada como: matrices_confusion.png");
        
        # Comparar diferentes topologías de RNA
        println("\nComparando diferentes topologías de RNA...");
        p3, topo_results, topo_names = compare_ann_topologies(dataset, cv_indices);
        savefig(p3, "comparacion_topologias.png");
        println("Gráfica de comparación de topologías guardada como: comparacion_topologias.png");
        
        # Mostrar curva de aprendizaje para la mejor topología
        best_topo_idx = argmax([r[1][1] for r in topo_results]);
        best_topology = [3, 5, 10, [5, 3], [10, 5]][best_topo_idx];
        println("\nGenerando curva de aprendizaje para la mejor topología: $(topo_names[best_topo_idx])");
        p4, _ = plot_learning_curve(best_topology, dataset, cv_indices);
        savefig(p4, "curva_aprendizaje.png");
        println("Curva de aprendizaje guardada como: curva_aprendizaje.png");
        
        # Mostrar todas las gráficas
        display(p1);
        display(p2);
        display(p3);
        display(p4);
        
    catch e
        println("Error durante la ejecución: ", e);
        println(stacktrace());
        return 1;
    end;
    
    return 0;
end;

# Ejecutar el programa principal
# exit(main());
compare_in_between()