dataset = readdlm("Practica2/optical+recognition+of+handwritten+digits/optdigits.full",',');
begin
    #Basic Hyperparameters
    # ----SVC---- ROTO
    # hyperparameters = Dict("C" => 1, "kernel" => "rbf", "gamma"  => 3);
    # modelType = :SVC;
    # ----DTC----Max = 0.695 depth = 9 cross nonorm
    hyperparameters = Dict("max_depth" => 9);
    modelType = :DecisionTreeClassifier;
    # ----KNN----Max = 0.968 K = 1 cross nonorm
    hyperparameters = Dict("n_neighbors" => 1);
    modelType = :KNeighborsClassifier;
    # ----ANN----
    # hyperparameters = Dict(
    #     "topology"        => [4, 3],
    #     "learningRate"    => 0.01,
    #     "validationRatio" => 0.2,
    #     "numExecutions"   => 50,
    #     "maxEpochs"       => 100,
    #     "maxEpochsVal"     => 20);
    # modelType = :ANN;

    #Dataset handling
 
    inputs = dataset[:,1:64];
    inputs = Float32.(inputs);
    targets = dataset[:,65]

    #Data normalization

    #inputs = normalizeMinMax(inputs)

    #training/measuring
    crossValidationIndices = crossvalidation(oneHotEncoding(targets), 10)
    # crossValidationIndices = repeat(1:10, 562)

    ((testAccuracy_mean, testAccuracy_std), (testErrorRate_mean, testErrorRate_std), (testRecall_mean, testRecall_std), (testSpecificity_mean, testSpecificity_std), (testPrecision_mean, testPrecision_std), (testNPV_mean, testNPV_std), (testF1_mean, testF1_std), testConfusionMatrix, best_model) =
    modelCrossValidation(modelType, hyperparameters, (inputs, targets), crossValidationIndices);

    #print data
    println(modelType, " ", hyperparameters)
    println("Media tasa de Acierto: ", testAccuracy_mean, " std: ", testAccuracy_std)
    println("Media tasa de Error: ", testErrorRate_mean, " std: ", testErrorRate_std)
    println("Media tasa de Sensibilidad: ", testRecall_mean, " std: ", testRecall_std)
    println("Media tasa de Especificidad: ", testSpecificity_std, " std: ", testSpecificity_std)
    println("Media tasa de precision: ", testPrecision_mean, " std: ", testPrecision_std)
    println("Media tasa de NPV: ", testNPV_mean, " std: ", testNPV_std)
    println("Media tasa de F1: ", testF1_mean, " std: ", testF1_std)
    println("Matriz de confusion: ", testConfusionMatrix)
    println("Mejor modelo: ", best_model[2])
    #Final prediction data
    model = best_model[1]
    pred = MLJ.predict(model, MLJ.table(inputs));
    predictions = mode.(pred);
    # print(accuracy(oneHotEncoding(predictions), oneHotEncoding(targets)))

end;