
# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses
using Random


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    if length(classes) > 2
        return convert(BitArray{2}, hcat([instance .== classes for instance in feature]...)');
    else
        return oneHotEncoding(convert(AbstractArray{Bool,1}, feature .== classes[1]));
    end;
end;

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    classes = convert(AbstractArray{<:Any, 1}, unique(feature));
    return oneHotEncoding(feature, classes);
end;

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, :, 1);
end;

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    min_col =  minimum(dataset, dims = 1);
    max_col = maximum(dataset, dims = 1);
    return (min_col, max_col);
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mean_col = mean(dataset, dims = 1);
    deviation_col = std(dataset, dims = 1);
    return (mean_col, deviation_col);
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    min_values, max_values = normalizationParameters;
    dataset .-= min_values;
    range_values = max_values .- min_values;

    dataset ./= (range_values);
    dataset[:, vec(min_values .== max_values)] .= 0;
    return dataset;
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset);
    return normalizeMinMax!(dataset, normalizationParameters);
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    new_dataset = copy(dataset);
    min_values, max_values = normalizationParameters;
    new_dataset .-= min_values;
    range_values = max_values .- min_values;

    new_dataset ./= (range_values);
    new_dataset[:, vec(min_values .== max_values)] .= 0;
    return new_dataset;
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset);
    normalizeMinMax(dataset, normalizationParameters);
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    mean_values, desviation_values = normalizationParameters;
    dataset .-= mean_values;
    dataset ./= desviation_values;
    dataset[:, vec(desviation_values .== 0)] .= 0;
    return dataset;
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset);
    return normalizeZeroMean!(dataset, normalizationParameters);
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    new_dataset = copy(dataset);
    mean_values, desviation_values = normalizationParameters;

    new_dataset .-= mean_values;
    new_dataset ./= desviation_values;
    new_dataset[:, vec(desviation_values .== 0)] .= 0;
    return new_dataset;
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset);
    return normalizeZeroMean(dataset, normalizationParameters);
end;

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .>= threshold;
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        outputs = classifyOutputs(outputs[:]; threshold);
        outputs = reshape(outputs, :, 1);
        return outputs;
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims = 2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true; 
        return outputs;
    end;
end;

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert size(targets) == size(outputs)
    return mean(targets .== outputs);
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert size(targets) == size(outputs)
    if size(outputs, 2) == 1

        return mean(targets[:, 1] .== outputs[:, 1]);
    else
        classComparison = targets .== outputs;
        correctClassifications = all(classComparison, dims = 2);
        accuracy = mean(correctClassifications);
        
        return accuracy;
    end;
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    @assert length(targets) == length(outputs)
    outputs = classifyOutputs(outputs; threshold);
    return accuracy(outputs,targets);
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert size(targets) == size(outputs)
    if size(outputs, 2) == 1
        return accuracy(outputs[:,1],targets[:,1]; threshold = threshold);
    else
        outputs = classifyOutputs(outputs; threshold);
        return accuracy(outputs,targets);
    end;
end;


function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    numInputsLayer = numInputs;
    ann = Chain();
    for (numOutputsLayer, transferFunction) in zip(topology, transferFunctions)
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunction));
        numInputsLayer = numOutputsLayer;
    end;
    if numOutputs > 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity), softmax);
    else
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    end;
    return ann;
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    ann = buildClassANN(size(dataset[1], 2), topology, size(dataset[2], 2), transferFunctions = transferFunctions);
    dataset = (permutedims(convert(AbstractArray{Float32, 2}, dataset[1])), permutedims(dataset[2]));
    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y);
    losses = [loss(ann, dataset[1], dataset[2])];
    cnt = 0;
    opt_state = Flux.setup(Adam(learningRate), ann);
    while cnt < maxEpochs && losses[length(losses)] > minLoss
        cnt += 1;
        Flux.train!(loss, ann, [dataset], opt_state);
        push!(losses, loss(ann, dataset[1], dataset[2]));
    end;
    return ann, losses;
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    dataset = reshape((inputs,targets), :, 1);
    trainClassANN(topology,dataset,transferFunctions = transferFunctions,maxEpochs = maxEpochs,minLoss = minLoss,learningRate = learningRate);
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

function holdOut(N::Int, P::Real)
    permutation = Random.randperm(N);
    test_index = permutation[1:(convert(Int64, (round(N*P))))];
    train_index = permutation[(convert(Int64, (round(N*P))) + 1):end];
    index = (train_index, test_index);
    return index;
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    train_index, test_index = holdOut(N, Ptest);
    
    new_N = length(train_index);
    new_Pval = Pval / (1 - Ptest);

    train_val = holdOut(new_N, new_Pval);

    new_train_index = train_index[train_val[1]];
    val_index = train_index[train_val[2]];

    return (new_train_index, val_index, test_index);
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    train_dataset = (permutedims(convert(AbstractArray{Float32, 2}, trainingDataset[1])), permutedims(trainingDataset[2]));

    ann = buildClassANN(size(trainingDataset[1], 2), topology, size(trainingDataset[2], 2), transferFunctions = transferFunctions);

    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y);
    
    train_losses = [loss(ann, train_dataset[1], train_dataset[2])];
    val_losses = Float32[];
    test_losses = Float32[];
    
    opt_state = Flux.setup(Adam(learningRate), ann);

    # Parada temprana
    best_val_loss = Inf32;
    best_epoch = 0;
    best_model = deepcopy(ann);

    has_validation = validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2));

    if has_validation
        valid_data = (permutedims(convert(AbstractArray{Float32,2},validationDataset[1])), permutedims(validationDataset[2]));
        push!(val_losses, loss(ann, valid_data[1], valid_data[2]));
        best_val_loss = val_losses[1];
    end;
    if testDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2))
        test_data = (permutedims(convert(AbstractArray{Float32,2},testDataset[1])), permutedims(testDataset[2]));
        push!(test_losses, loss(ann, test_data[1], test_data[2]));
    end;

    cnt = 0;
    while cnt < maxEpochs && train_losses[end] > minLoss && ((!has_validation) || (cnt - best_epoch < maxEpochsVal))
        cnt += 1;
        Flux.train!(loss, ann, [train_dataset], opt_state);
        push!(train_losses, loss(ann, train_dataset[1], train_dataset[2]));

        if has_validation
            val_loss = loss(ann, valid_data[1], valid_data[2]);
            push!(val_losses, val_loss);

            if val_loss < best_val_loss
                best_val_loss = val_loss;
                best_epoch = cnt;
                best_model = deepcopy(ann);
            end;
        end;

        if testDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2))
            push!(test_losses, loss(ann, test_data[1], test_data[2]));
        end;
    end;

    if has_validation
        return best_model, train_losses, val_losses, test_losses;
    else
        return ann, train_losses, val_losses, test_losses;
    end;
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)

    trainingDataset = (trainingDataset[1],reshape(trainingDataset[2], :, 1));
    validationDataset = (validationDataset[1],reshape(validationDataset[2], :, 1));
    testDataset = (testDataset[1],reshape(testDataset[2], :, 1));
    
    trainClassANN(topology,trainingDataset,validationDataset=validationDataset,testDataset=testDataset,transferFunctions = transferFunctions,maxEpochs = maxEpochs,minLoss = minLoss,learningRate = learningRate,maxEpochsVal = maxEpochsVal);

end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum(outputs .&& targets);
    VN = sum(.!outputs .&& .!targets);
    FP = sum(outputs .&& .!targets);
    FN = sum(.!outputs .&& targets);

    matriz_confusion = [VP FP; FN VN];

    precision = (VN + VP) / (VN + VP + FN + FP);
    tasa_error = (FN + FP) / (VN + VP + FN + FP);
    sensibilidad = VP / (FN + VP);
    especificidad = VN / (FP + VN);
    valor_predictivo_positivo = VP / (VP + FP);
    valor_predictivo_negativo = VN / (VN + FN);
    f1_score = (2*valor_predictivo_positivo*sensibilidad)/ (valor_predictivo_positivo + sensibilidad);

    if VP == 0 && FN == 0  
        sensibilidad = 1;
    end;
    if VP == 0 && FP == 0  
        valor_predictivo_positivo = 1;
    end;
    if FN == 0 && FP == 0  
        especificidad = 1;
    end;
    if VN == 0 && FN == 0  
        valor_predictivo_negativo = 1  ;
    end;

    if valor_predictivo_positivo == 0 && sensibilidad == 0
        f1_score = 0;
    end;
    return (precision, tasa_error, sensibilidad, especificidad, valor_predictivo_positivo, valor_predictivo_negativo, f1_score, matriz_confusion)
    

end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    new_outputs = classifyOutputs(outputs, threshold = threshold);
    confusionMatrix(new_outputs, targets);
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    
    if (size(outputs, 2) != size(targets, 2)) && size(outputs, 2) == 1
        return confusionMatrix(outputs[:,1], targets[:,1], strategy);
    end;

    num_classes = size(outputs, 2);
    sensibilidad = zeros(Float64, num_classes);
    especificidad = zeros(Float64, num_classes);
    valor_predictivo_positivo = zeros(Float64, num_classes);
    valor_predictivo_negativo = zeros(Float64, num_classes);
    f1_score = zeros(Float64, num_classes);

    for i in 1:num_classes
        outputs_class = outputs[:, i];
        targets_class = targets[:, i];
        
        stats = confusionMatrix(outputs_class, targets_class);
        sensibilidad, especificidad, valor_predictivo_positivo, valor_predictivo_negativo, f1_score = stats[3:end];
    end;

    matriz_confusion = [sum((outputs .== i) .&& (targets .== j)) for i in 1:num_classes, j in 1:num_classes];

    instancias_clase = vec(sum(targets, dims=1));

    if weighted == true
        sensibilidad_media = sum(sensibilidad .* instancias_clase) / sum(instancias_clase);
        especificidad_media = sum(especificidad .* instancias_clase) / sum(instancias_clase);
        valor_predictivo_positivo_medio = sum(valor_predictivo_positivo .* instancias_clase) / sum(instancias_clase);
        valor_predictivo_negativo_medio = sum(valor_predictivo_negativo .* instancias_clase) / sum(instancias_clase);
        f1_score_medio = sum(f1_score .* instancias_clase) / sum(instancias_clase);

    else
        sensibilidad_media = mean(sensibilidad);
        especificidad_media = mean(especificidad);
        valor_predictivo_positivo_medio = mean(valor_predictivo_positivo);
        valor_predictivo_negativo_medio = mean(valor_predictivo_negativo);
        f1_score_medio = mean(f1_score1);
        
    end;

    precision = accuracy(outputs, targets);
    tasa_error = 1 - precision;

    return (precision, tasa_error, sensibilidad, especificidad, valor_predictivo_positivo, valor_predictivo_negativo, f1_score, matriz_confusion);
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    new_outputs = classifyOutputs(outputs, threshold = threshold);
    confusionMatrix(new_outputs, targets, weighted = weighted);
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(label, classes) for label in vcat(targets, outputs)]));
    bool_outputs = oneHotEncoding(outputs, classes);
    bool_targets = oneHotEncoding(targets, classes);
    return confusionMatrix(bool_outputs, bool_targets, weighted = weighted);
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs));
    return confusionMatrix(outputs, targets, classes, weighted = weighted);
end;

using SymDoME


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = convert(AbstractArray{Float64,2}, trainingDataset[1]);
    trainingTargets = convert(AbstractArray{Bool,1}, trainingDataset[2]);
    testInputs = convert(AbstractArray{Float64,2}, testInputs);

    _, _, _, model = dome(trainingInputs, trainingTargets; maximumNodes = maximumNodes);

    return evaluateTree(model, testInputs);
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = convert(AbstractArray{Float64,2}, trainingDataset[1]);
    trainingTargets = convert(AbstractArray{Bool,2}, trainingDataset[2]);
    testInputs = convert(AbstractArray{Float64,2}, testInputs);

    if size(trainingTargets,2) == 1
        result = trainClassDoME((trainingInputs, vec(trainingTargets)), testInputs, maximumNodes);
        return reshape(result, :, 1);
    end;

    num_classes = size(trainingTargets, 2);
    result = zeros(Float64, size(testInputs,1), num_classes);

    for i in 1:num_classes
        class_results = trainClassDoME((trainingInputs, trainingTargets[:,i]), testInputs, maximumNodes);

        result[:,i] = class_results;
    end;

    return result
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = convert(AbstractArray{Float64,2}, trainingDataset[1]);
    trainingTargets = trainingDataset[2];
    testInputs = convert(AbstractArray{Float64,2}, testInputs);

    classes = unique(trainingTargets);

    testOutputs = Array{eltype(trainingTargets),1}(undef, size(testInputs, 1));
    testOutputsDoME = trainClassDoME((trainingInputs, oneHotEncoding(trainingTargets, classes)), testInputs, maximumNodes);
    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0);

    num_classes = length(classes);
    
    if num_classes <= 2
        testOutputsBool = vec(testOutputsBool);
        testOutputs[testOutputsBool] .= classes[1];

        if num_classes == 2
            testOutputs[.!testOutputsBool] .= classes[2];
        end;
    else
        for i in 1:num_classes
            testOutputs[testOutputsBool[:,i]] .= classes[i];
        end;
    end;

    return testOutputs;
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1},
    targets::AbstractArray{Bool,1})
    ann = confusionMatrix(outputs,targets);
    print("Valor de precisión: ",ann[1]);
    print("Tasa de fallo: ", ann[2]);
    print("Sensibilidad: ", ann[3]);
    print("Especificidad: ", ann[4]);
    print("Valor predictivo positivo: ", ann[5]);
    print("F1-score: ", ann[6]);
    print("Matriz de confusión: ", ann[7]);

end;
    

function printConfusionMatrix(outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1}; threshold::Real=0.5) 
    ann = confusionMatrix(outputs, targets, threshold=threshold);
    print("Valor de precisión: ",ann[1]);
    print("Tasa de fallo: ", ann[2]);
    print("Sensibilidad: ", ann[3]);
    print("Especificidad: ", ann[4]);
    print("Valor predictivo positivo: ", ann[5]);
    print("F1-score: ", ann[6]);
    print("Matriz de confusión: ", ann[7]);

end;


function printConfusionMatrix(outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
    ann = confusionMatrix(outputs, targets, weighted);
    print("Valor de precisión: ",ann[1]);
    print("Tasa de fallo: ", ann[2]);
    print("Sensibilidad: ", ann[3]);
    print("Especificidad: ", ann[4]);
    print("Valor predictivo positivo: ", ann[5]);
    print("F1-score: ", ann[6]);
    print("Matriz de confusión: ", ann[7]);
end;
    
    
function printConfusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
    ann = confusionMatrix(outputs, targets, weighted);
    print("Valor de precisión: ",ann[1]);
    print("Tasa de fallo: ", ann[2]);
    print("Sensibilidad: ", ann[3]);
    print("Especificidad: ", ann[4]);
    print("Valor predictivo positivo: ", ann[5]);
    print("F1-score: ", ann[6]);
    print("Matriz de confusión: ", ann[7]);

end;


function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1},
    classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(label, classes) for label in vcat(targets, outputs)]));
    bool_outputs = oneHotEncoding(outputs, classes);
    bool_targets = oneHotEncoding(targets, classes);
    ann = confusionMatrix(bool_outputs, bool_targets, weighted);
    print("Valor de precisión: ",ann[1]);
    print("Tasa de fallo: ", ann[2]);
    print("Sensibilidad: ", ann[3]);
    print("Especificidad: ", ann[4]);
    print("Valor predictivo positivo: ", ann[5]);
    print("F1-score: ", ann[6]);
    print("Matriz de confusión: ", ann[7]);

end;


function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs));
    ann = confusionMatrix(outputs, classes, weighted);
    print("Valor de precisión: ",ann[1]);
    print("Tasa de fallo: ", ann[2]);
    print("Sensibilidad: ", ann[3]);
    print("Especificidad: ", ann[4]);
    print("Valor predictivo positivo: ", ann[5]);
    print("F1-score: ", ann[6]);
    print("Matriz de confusión: ", ann[7]);


end;

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    v = 1:N;
    v_repeated = repeat(v, convert(Int64, round(ceil(N/k))));
    v_sliced = v_repeated[1:N];
    v_random = Random.shuffle!(v_sliced);
    return v_random;
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    if k < 10
        print("ERROR, k < 10");
        return
    end;
    v = 1:length(targets);
    num_true = sum(targets);
    cross_index = crossvalidation(num_true, k);
    true_positions = findall(targets);
    v[true_positions] .= cross_index;
    return v
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    if k < 10
        print("ERROR, k < 10");
        return
    end;    
    v = 1:size(targets, 1);
    for j in eachcol(targets)
        num_true = sum(j);
        cross_index = crossvalidation(num_true, k);
        true_positions = findall(targets);
        v[true_positions] .= cross_index;
    end
    return v
    
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    targets_bool = oneHotEncoding(targets)
    v = crossvalidation(targets_bool, k)
    return v
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

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
            model = SVMClassifier(C=modelHyperparameters["C"], kernel=modelHyperparameters["kernel"],
            degree=modelHyperparameters["degree"], gamma=modelHyperparameters["gamma"],
            coef0=modelHyperparameters["coef0"]);
        elseif modelType == :DecisionTreeClassifier
            model = DTClassifier(max_depth=modelHyperparameters["max_depth"], random_state=1);
        elseif modelType == :KNeighborsClassifier
            model = kNNClassifier(n_neighbors=modelHyperparameters["n_neighbors"]);
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