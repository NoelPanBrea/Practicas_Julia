using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
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
        else
            error("Hiperparámetro 'topology' no encontrado en 'modelHyperparameters'.");
        end;
    end;

    precision, errorRate, sensitivity, specificity, ppv, npv, f1 = ([] for _ in 1:7);

    targets = string.(targets);
    classes = unique(targets);

    for fold in unique(crossValidationIndices)
        trainIndices = findall(x -> x != fold, crossValidationIndices);
        testIndices = findall(x -> x == fold, crossValidationIndices);

        X_train, X_test = inputs[trainIndices, :], inputs[testIndices, :];
        y_train, y_test = targets[trainIndices], targets[testIndices];

        if modelType == :SVC
            model = SVMClassifier(kernel=LIBSVM.Kernel.RadialBasis,
            cost=Float64(modelHyperparameters["C"]), gamma=Float64(modelHyperparameters["gamma"]));
        elseif modelType == :DecisionTreeClassifier
            model = DTClassifier(max_depth=Float64(modelHyperparameters["max_depth"]), random_state=1);
        elseif modelType == :KNeighborsClassifier
            model = kNNClassifier(n_neighbors=Float64(modelHyperparameters["n_neighbors"]));
        end;
        
        fit!(model, X_train, y_train);
    
        predictions = predict(model, X_test);
       
        acc, fail_rate, sensitivities, specificities, ppvs, npvs, f1s = confusionMatrix(y_test, predictions);

        push!(precision, acc);
        push!(errorRate, fail_rate);
        push!(sensitivity, sensitivities);
        push!(specificity, specificities);
        push!(ppv, ppvs);
        push!(npv, npvs);
        push!(f1, f1s);  
    end;

    result = ((mean(precision), std(precision)), (mean(errorRate), std(errorRate)),
              (mean(sensitivity), std(sensitivity)), (mean(specificity), std(specificity)),
              (mean(ppv), std(ppv)), (mean(npv), std(npv)), (mean(f1), std(f1)));
    return result;
end;