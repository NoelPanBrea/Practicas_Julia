function TrainCrossValEnsembles(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{DataFrame, BitMatrix}, crossValidationIndices::Array{Int64,1})
    accuracy = Float64[];
    sensitivity = Float64[];
    ppv = Float64[];
    f1 = Float64[];
    inputs, targets = dataset;
    for fold in unique(crossValidationIndices)
        
        trainIndices = findall(x -> x != fold, crossValidationIndices);
        testIndices = findall(x -> x == fold, crossValidationIndices);

        train_inputs = pca(inputs[trainIndices, :], 0.95)[1];
        train_targets = categorical(sum.(findall.(eachrow(targets[trainIndices, :]))), ordered = true, levels=[1, 2, 3, 4, 5, 6]);
        test_inputs = pca(inputs[testIndices, :], n_components=size(train_inputs, 2))[1];
        test_targets = categorical(sum.(findall.(eachrow(targets[testIndices, :]))), ordered = true, levels=[1, 2, 3, 4, 5, 6]);

        if modelType == :AdaB
            model = AdaBoost(n_iter=modelHyperparameters["n_iter"])
            mach = machine(model, train_inputs, train_targets, scitype_check_level=0);
            MLJ.fit!(mach, verbosity=0);
            pred = MLJ.predict(mach, test_inputs);
            predictions = mode.(pred);
        elseif modelType == :BaggingC
            model = MLJ.EnsembleModel(model=kNNClassifier(K=5),
                                    bagging_fraction=0.5,
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

function TrainEnsembles(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{Tuple{DataFrame, BitMatrix}, Tuple{DataFrame, BitMatrix}})
    ((train_inputs, train_targets), (test_inputs, test_targets)) = dataset;

    train_targets = categorical(sum.(findall.(eachrow(train_targets))), ordered = true, levels=[1, 2, 3, 4, 5, 6]);
    test_targets = categorical(sum.(findall.(eachrow(test_targets))), ordered = true, levels=[1, 2, 3, 4, 5, 6]);

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
                               eta=0.2);

            mach = machine(model, train_inputs, train_targets);
            MLJ.fit!(mach, verbosity=0);
            pred = MLJ.predict(mach, test_inputs);
            predictions = mode.(pred);
        elseif modelType == :XGB
            model = XGBoostClassifier();

            mach = machine(model, train_inputs, train_targets);
            MLJ.fit!(mach, verbosity=0);
            pred = MLJ.predict(mach, test_inputs);
            predictions = mode.(pred);
        elseif modelType == :CatB
            model = CatBoostClassifier();

            mach = machine(model, train_inputs, train_targets);
            MLJ.fit!(mach, verbosity=0);
            pred = MLJ.predict(mach, test_inputs);
            predictions = mode.(pred);
        elseif modelType == :LGBM
            model = LGBMClassifier();

            mach = machine(model, train_inputs, train_targets);
            MLJ.fit!(mach, verbosity=0);
            pred = MLJ.predict(mach, test_inputs);
            predictions = mode.(pred);
        end;

        accuracy = MLJ.accuracy(predictions, test_targets)
        sensitivity = MLJ.sensitivity(predictions, test_targets)
        ppv = MLJ.ppv(predictions, test_targets)
        f1score = MLJ.f1score(predictions, test_targets)

    return accuracy, sensitivity, ppv, f1score;
end; 