using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
    # Extraer datos del dataset
    inputs, targets = dataset;
    
    # Si el modelo es una Red Neuronal Artificial, llamar a la función correspondiente
    if modelType == :ANN
        if haskey(modelHyperparameters, "topology")
            topology = modelHyperparameters["topology"];
            
            return ANNCrossValidation(topology, inputs, targets, crossValidationIndices;
                numExecutions = get(modelHyperparameters, "numExecutions", 50),
                transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(topology))),
                maxEpochs = get(modelHyperparameters, "maxEpochs", 1000),
                minLoss = get(modelHyperparameters, "minLoss", 0.0),
                learningRate = get(modelHyperparameters, "learningRate", 0.01),
                validationRatio = get(modelHyperparameters, "validationRatio", 0),
                maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", 20));
        end;
    end;

    # Vectores para almacenar métricas de cada fold
    accuracy = Float64[];
    error_rate = Float64[];
    sensitivity = Float64[];
    specificity = Float64[];
    ppv = Float64[];
    npv = Float64[];
    f1 = Float64[];

    # Convertir el vector de salidas a strings
    targets = string.(targets);
    
    # Calcular las clases únicas
    classes = unique(targets);
    
    # Inicializar la matriz de confusión
    confusion_matrix = zeros(Int64, length(classes), length(classes));
    
    # Para cada fold en la validación cruzada
    for fold in unique(crossValidationIndices)
        # Índices para entrenamiento y test
        trainIndices = findall(x -> x != fold, crossValidationIndices);
        testIndices = findall(x -> x == fold, crossValidationIndices);
        
        # Extraer datos de entrenamiento y test
        X_train = inputs[trainIndices, :];
        y_train = targets[trainIndices];
        X_test = inputs[testIndices, :];
        y_test = targets[testIndices];
        
        # Variable para almacenar predicciones
        predictions = nothing;
        
        if modelType == :DoME
            # Para DoME, usamos la función trainClassDoME
            maximumNodes = modelHyperparameters["maximumNodes"];
            predictions = trainClassDoME((X_train, y_train), X_test, maximumNodes);
            
        elseif modelType == :SVC
            # Para SVM, configuramos según el tipo de kernel
            kernel_type = modelHyperparameters["kernel"];
            C = modelHyperparameters["C"];
            
            if kernel_type == "linear"
                model = SVMClassifier(
                    kernel = LIBSVM.Kernel.Linear,
                    cost = Float64(C)
                )
            elseif kernel_type == "rbf"
                gamma = modelHyperparameters["gamma"];
                model = SVMClassifier(
                    kernel = LIBSVM.Kernel.RadialBasis,
                    cost = Float64(C),
                    gamma = Float64(gamma)
                )
            elseif kernel_type == "sigmoid"
                gamma = modelHyperparameters["gamma"];
                coef0 = modelHyperparameters["coef0"];
                model = SVMClassifier(
                    kernel = LIBSVM.Kernel.Sigmoid,
                    cost = Float64(C),
                    gamma = Float64(gamma),
                    coef0 = Float64(coef0)
                )
            elseif kernel_type == "poly"
                gamma = modelHyperparameters["gamma"];
                coef0 = modelHyperparameters["coef0"];
                degree = modelHyperparameters["degree"];
                model = SVMClassifier(
                    kernel = LIBSVM.Kernel.Polynomial,
                    cost = Float64(C),
                    gamma = Float64(gamma),
                    coef0 = Float64(coef0),
                    degree = Int32(degree)
                )
            end
            
            # Crear machine, entrenar y predecir
            mach = machine(model, MLJ.table(X_train), categorical(y_train));
            MLJ.fit!(mach, verbosity=0);
            predictions = MLJ.predict(mach, MLJ.table(X_test));
            
        elseif modelType == :DecisionTreeClassifier
            # Para árboles de decisión
            max_depth = modelHyperparameters["max_depth"];
            model = DTClassifier(max_depth = max_depth, rng = Random.MersenneTwister(1));
            
            mach = machine(model, MLJ.table(X_train), categorical(y_train));
            MLJ.fit!(mach, verbosity=0);
            pred = MLJ.predict(mach, MLJ.table(X_test));
            predictions = mode.(pred);
            
        elseif modelType == :KNeighborsClassifier
            # Para kNN
            K = modelHyperparameters["n_neighbors"];
            model = kNNClassifier(K = K);
            
            mach = machine(model, MLJ.table(X_train), categorical(y_train));
            MLJ.fit!(mach, verbosity=0);
            pred = MLJ.predict(mach, MLJ.table(X_test));
            predictions = mode.(pred);
        end
        
        # Calcular métricas y matriz de confusión para este fold
        fold_metrics = confusionMatrix(predictions, y_test, classes);
        
        # Almacenar métricas
        push!(accuracy, fold_metrics[1]);
        push!(error_rate, fold_metrics[2]);
        push!(sensitivity, fold_metrics[3]);
        push!(specificity, fold_metrics[4]);
        push!(ppv, fold_metrics[5]);
        push!(npv, fold_metrics[6]);
        push!(f1, fold_metrics[7]);
        
        # Actualizar matriz de confusión global
        confusion_matrix += fold_metrics[8];
    end
    
    # Calcular estadísticas para cada métrica
    accuracy_stats = (mean(accuracy), std(accuracy));
    error_rate_stats = (mean(error_rate), std(error_rate));
    sensitivity_stats = (mean(sensitivity), std(sensitivity));
    specificity_stats = (mean(specificity), std(specificity));
    ppv_stats = (mean(ppv), std(ppv));
    npv_stats = (mean(npv), std(npv));
    f1_stats = (mean(f1), std(f1));
    
    # Devolver las métricas y la matriz de confusión
    return accuracy_stats, error_rate_stats, sensitivity_stats, specificity_stats, 
           ppv_stats, npv_stats, f1_stats, confusion_matrix;
end