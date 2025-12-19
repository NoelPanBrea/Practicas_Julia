function TrainBaseEnsembles(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{DataFrame, BitMatrix}, crossValidationIndices::Array{Int64,1}, k::Int)
    accuracy = Float64[];
    sensitivity = Float64[];
    ppv = Float64[];
    f1 = Float64[];
    inputs, targets = dataset;
    for fold in unique(crossValidationIndices)
        
        trainIndices = findall(x -> x != fold, crossValidationIndices);
        testIndices = findall(x -> x == fold, crossValidationIndices);

        train_inputs = pca(inputs[trainIndices, :], n_components=k)[1];
        train_targets = categorical(sum.(findall.(eachrow(targets[trainIndices, :]))));
        test_inputs = pca(inputs[testIndices, :], n_components=k)[1];
        test_targets = categorical(sum.(findall.(eachrow(targets[testIndices, :]))));


        if modelType == :AdaB
            model = AdaBoost(n_iter=modelHyperparameters["n_iter"])
            mach = machine(model, train_inputs, train_targets, scitype_check_level=0);
            MLJ.fit!(mach, verbosity=0);
            pred = MLJ.predict(mach, test_inputs);
            predictions = mode.(pred);
        elseif modelType == :BaggingC
            model = MLJ.EnsembleModel(model=kNNClassifier(K=5),
                                    bagging_fraction=1,
                                    n=modelHyperparameters["n_estimators"])
            mach = machine(model, train_inputs, train_targets, scitype_check_level=0);
            MLJ.fit!(mach, verbosity=0);
            pred = MLJ.predict(mach, test_inputs);
            predictions = mode.(pred);

        elseif modelType == :EvoT
            model = EvoTreeClassifier(nrounds=modelHyperparameters["n_estimators"],
                               eta=0.2)

            mach = machine(model, train_inputs, train_targets);
            MLJ.fit!(mach, verbosity=0);
            pred = MLJ.predict(mach, test_inputs);
            predictions = mode.(pred);
        end

        push!(accuracy, MLJ.accuracy(predictions, test_targets));
        push!(sensitivity, MLJ.sensitivity(predictions, test_targets));
        push!(ppv, MLJ.ppv(predictions, test_targets));
        push!(f1, MLJ.f1score(predictions, test_targets));

    end;

    accuracy_stats = (mean(accuracy), std(accuracy));
    sensitivity_stats = (mean(sensitivity), std(sensitivity));
    ppv_stats = (mean(ppv), std(ppv));
    f1_stats = (mean(f1), std(f1));
    return accuracy_stats, sensitivity_stats, ppv_stats, f1_stats;
end; 