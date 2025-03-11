using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
    #=
     SVMClassifier(kernel = LIBSVM.Kernel.Polynomial, cost = Float64(C),
            gamma = Float64(gamma), degree = Int32(degree), coef0 = Float64(coef0));
    PDF modelType := SVC |
    =#
    if modelType == :ANN
        # Asegúrate de que 'topology' existe dentro de 'modelHyperparameters'
        if haskey(modelHyperparameters, "topology")
            topology = modelHyperparameters["topology"];
            # Llamada ajustada a ANNCrossValidation con el argumento esperado
            return ANNCrossValidation(topology, inputs, targets, crossValidationIndices;
                numExecutions = get(modelHyperparameters, "numExecutions", 50),
                transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(topology))),
                maxEpochs = get(modelHyperparameters, "maxEpochs", 1000),
                minLoss = get(modelHyperparameters, "minLoss", 0.0),
                learningRate = get(modelHyperparameters, "learningRate", 0.01),
                validationRatio = get(modelHyperparameters, "validationRatio", 0),
                maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", 20));
        else
            error("Hiperparámetro 'topology' no encontrado en 'modelHyperparameters'.");
        end;
    end;

    # Inicialización de los 7 vectores para resultados de métricas
    precision, errorRate, sensitivity, specificity, ppv, npv, f1 = ([] for _ in 1:7);

    # Conversión de targets a string para compatibilidad con Scikit-Learn
    targets = string.(targets);
    classes = unique(targets); #No sé para que hay que usarlo, pero lo pone en el PDF

    #BUCLE DE CROSSVALIDACIÓN | Falta la parte de DoME
    for fold in unique(crossValidationIndices)
        trainIndices = findall(x -> x != fold, crossValidationIndices);
        testIndices = findall(x -> x == fold, crossValidationIndices);

        X_train, X_test = inputs[trainIndices, :], inputs[testIndices, :];
        y_train, y_test = targets[trainIndices], targets[testIndices];

        # Creación y entrenamiento del modelo (primero tienen que ser creados y luego entrenados)
        # Entrenamiento DoME: TrainDoMe()
        if modelType == :SVC
            model = SVMClassifier(C=modelHyperparameters["C"], kernel=modelHyperparameters["kernel"],
            degree=modelHyperparameters["degree"], gamma=modelHyperparameters["gamma"],
            coef0=modelHyperparameters["coef0"]);
        elseif modelType == :DecisionTreeClassifier
            model = DTClassifier(max_depth=modelHyperparameters["max_depth"], random_state=1);
        elseif modelType == :KNeighborsClassifier
            model = kNNClassifier(n_neighbors=modelHyperparameters["n_neighbors"]);
        end;
        # Entrenamiento del modelo con los conjuntos de entrenamiento
        fit!(model, X_train, y_train);
        # En el PDF pone que hay que usar machine, investigar para qué!

        # Realización de predicciones con el conjunto de prueba
        predictions = predict(model, X_test);
        # Calculo de métricas con las predicciones y ly_test))ts de prueba
        acc, fail_rate, sensitivities, specificities, ppvs, npvs, f1s = confusionMatrix(y_test, predictions);

        # Almacenamiento de resultados de métricas
        push!(precision, acc);
        push!(errorRate, fail_rate);
        push!(sensitivity, sensitivities);
        push!(specificity, specificities);
        push!(ppv, ppvs);
        push!(npv, npvs);
        push!(f1, f1s);  
    end;

    # Cálculo de la media y desviación estándar de cada métrica | Falta añadir una cosa CREO
    result = ((mean(precision), std(precision)), (mean(errorRate), std(errorRate)),
              (mean(sensitivity), std(sensitivity)), (mean(specificity), std(specificity)),
              (mean(ppv), std(ppv)), (mean(npv), std(npv)), (mean(f1), std(f1)));
    return result;
end;