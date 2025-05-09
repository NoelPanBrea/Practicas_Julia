
# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses


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

    matriz_confusion = [VN FP; FN VP];

    if VP == 0 && FN == 0  
        sensibilidad = 1;
    else
        sensibilidad = VP / (FN + VP);
    end;
    
    if VP == 0 && FP == 0  
        valor_predictivo_positivo = 1;
    else
        valor_predictivo_positivo = VP / (VP + FP);
    end;

    if VN == 0 && FP == 0  
        especificidad = 1;
    else
        especificidad = VN / (FP + VN);
    end;

    if VN == 0 && FN == 0  
        valor_predictivo_negativo = 1  ;
    else
        valor_predictivo_negativo = VN / (VN + FN);
    end;

    precision = (VN + VP) / (VN + VP + FN + FP);
    tasa_error = (FN + FP) / (VN + VP + FN + FP);

    if valor_predictivo_positivo == 0 && sensibilidad == 0
        f1_score = 0;
    else
        f1_score = (2*valor_predictivo_positivo*sensibilidad)/ (valor_predictivo_positivo + sensibilidad);
    end;

    return (precision, tasa_error, sensibilidad, especificidad, valor_predictivo_positivo, valor_predictivo_negativo, f1_score, matriz_confusion)
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    new_outputs = classifyOutputs(outputs, threshold = threshold);
    confusionMatrix(new_outputs, targets);
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    if size(outputs, 2) == 1 && size(targets, 2) == 1
        return confusionMatrix(outputs[:,1], targets[:,1]);
    end;

    if size(outputs, 2) > 2 && size(targets, 2) > 2
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
            _, _, sensibilidad[i], especificidad[i], valor_predictivo_positivo[i], 
            valor_predictivo_negativo[i], f1_score[i] = stats
        end;
  
        matriz_confusion = [sum((outputs[:, i] .== 1) .&& (targets[:, j] .== 1)) 
                            for i in 1:num_classes, j in 1:num_classes]
        matriz_confusion = permutedims(matriz_confusion);
        
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
            f1_score_medio = mean(f1_score);
        end;

        precision = accuracy(outputs, targets);
        tasa_error = 1 - precision;

        return (precision, tasa_error, sensibilidad_media, especificidad_media, valor_predictivo_positivo_medio, valor_predictivo_negativo_medio, f1_score_medio, matriz_confusion);
    end;
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    new_outputs = classifyOutputs(outputs, threshold = threshold);
    return confusionMatrix(new_outputs, targets, weighted = weighted);
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(label, classes) for label in vcat(targets, outputs)]));
    bool_outputs = oneHotEncoding(outputs);
    bool_targets = oneHotEncoding(targets);
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

    testOutputs = evaluateTree(model, testInputs);
    #Se añade el caso de que puede ser un valor Real o un vector
    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(testInputs, 1));
    end;
    return testOutputs;
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = convert(AbstractArray{Float64,2}, trainingDataset[1]);
    trainingTargets = convert(AbstractArray{Bool,2}, trainingDataset[2]);
    num_classes = size(trainingDataset[2],2);

    if num_classes == 1
        result = trainClassDoME((trainingInputs, vec(trainingTargets)), testInputs, maximumNodes);
        return reshape(result, :, 1);
    end;

    @assert(num_classes>2);

    result = Array{Float64,2}(undef, size(testInputs,1), num_classes);

    for i in Base.OneTo(num_classes)

        result[:,i] .= trainClassDoME((trainingInputs, trainingTargets[:,i]), testInputs, maximumNodes);

    end;

    return result
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    (trainingInputs, trainingTargets) = trainingDataset;
    classes = unique(trainingTargets);

    testOutputs = Array{eltype(trainingTargets),1}(undef, size(testInputs, 1));

    testOutputsDoME = trainClassDoME((trainingInputs, oneHotEncoding(trainingTargets, classes)), testInputs, maximumNodes);
    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0);

    num_classes = length(classes);
    
    if num_classes <= 2
        @assert(isa(testOutputsBool, Vector) || size(testOutputsBool,2)==1)
        testOutputsBool = vec(testOutputsBool);
        testOutputs[testOutputsBool] .= classes[1];

        if num_classes == 2
            testOutputs[.!testOutputsBool] .= classes[2];
        else @assert(all(testOutputsBool))
        end;
    else 
        @assert(all(sum(testOutputsBool, dims=2).==1));
        for i in eachindex(classes)
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
    v = 1:k;
    v_repeated = repeat(v, convert(Int64, ceil(N/k)));
    v_sliced = v_repeated[1:N];
    return Random.shuffle!(v_sliced);
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    len = length(targets);

    if k < 1
        error("ERROR: k debe ser al menos 1");
    end;
    
    v = zeros(Int64, len);
    
    num_true = sum(targets);
    num_false = len - num_true;

    if (num_true < k & num_true < 10) || (num_false < k & num_false < 10)
        error("Error: no hay suficientes patrones de cada clase");
    end;

    true_positions = findall(targets);
    false_positions = findall(.!targets);

    v[true_positions] .= crossvalidation(num_true, k);
    v[false_positions] .= crossvalidation(num_false, k);
    #Para hacerlo en una línea foreach(((pos, n),) -> v[pos] .= crossvalidation(n, k), [(true_positions, num_true), (false_positions, num_false)])
    return v
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    num_patterns = size(targets, 1);
    num_classes = size(targets,2);

    if k < 1
        error("ERROR: k debe ser al menos 1");
    end;

    v = zeros(Int64, num_patterns);

    for j in 1:num_classes
        class_positions = findall(targets[:,j]);

        num_class_patterns = length(class_positions)
        if num_class_patterns < k & num_class_patterns < 10
            error("Error no hay suficientes patrones para la clase $j");
        end;

        v[class_positions] = crossvalidation(num_class_patterns,k);
    end;
    return v
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    targets_bool = oneHotEncoding(targets);
    return crossvalidation(targets_bool, k)
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    
    (inputs,targets) = dataset;

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));

    # Que clases de salida tenemos
    classes = unique(targets);

    # Primero codificamos las salidas deseadas en caso de entrenar RR.NN.AA.
    targets = oneHotEncoding(targets, classes);

    # Creamos los vectores para las metricas que se vayan a usar
    numFolds = maximum(crossValidationIndices);
    testAccuracy    = Array{Float64,1}(undef, numFolds);
    testErrorRate   = Array{Float64,1}(undef, numFolds);
    testRecall      = Array{Float64,1}(undef, numFolds);
    testSpecificity = Array{Float64,1}(undef, numFolds);
    testPrecision   = Array{Float64,1}(undef, numFolds);
    testNPV         = Array{Float64,1}(undef, numFolds);
    testF1          = Array{Float64,1}(undef, numFolds);
    testConfusionMatrix = zeros(length(classes), length(classes));

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Dividimos los datos en entrenamiento y test
        trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
        testInputs        = inputs[crossValidationIndices.==numFold,:];
        trainingTargets   = targets[crossValidationIndices.!=numFold,:];
        testTargets       = targets[crossValidationIndices.==numFold,:];

        # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
        #  se crean vectores adicionales para almacenar las metricas para cada entrenamiento
        testAccuracyEachRepetition        = Array{Float64,1}(undef, numExecutions);
        testErrorRateEachRepetition       = Array{Float64,1}(undef, numExecutions);
        testRecallEachRepetition          = Array{Float64,1}(undef, numExecutions);
        testSpecificityEachRepetition     = Array{Float64,1}(undef, numExecutions);
        testPrecisionEachRepetition       = Array{Float64,1}(undef, numExecutions);
        testNPVEachRepetition             = Array{Float64,1}(undef, numExecutions);
        testF1EachRepetition              = Array{Float64,1}(undef, numExecutions);
        testConfusionMatrixEachRepetition = Array{Float64,3}(undef, length(classes), length(classes), numExecutions);

        # Se entrena las veces que se haya indicado
        for numTraining in 1:numExecutions

            if validationRatio>0

                # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                #  Para ello, hacemos un hold out
                (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*size(inputs,1)/size(trainingInputs,1));
                # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA
                # Otra forma de hacer el mismo cálculo:
                # (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), validationRatio*numFolds/(numFolds-1));

                # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                ann, = trainClassANN(topology, (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                    validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                    testDataset =       (testInputs,                          testTargets);
                    transferFunctions = transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal);
                    
            else

                # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                #  teniendo cuidado de codificar las salidas deseadas correctamente
                ann, = trainClassANN(topology, (trainingInputs, trainingTargets),
                    testDataset = (testInputs,     testTargets);
                    transferFunctions=transferFunctions,
                    maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate);

            end;

            # Calculamos las metricas correspondientes con la funcion desarrollada en el ejercicio anterior
            (testAccuracyEachRepetition[numTraining], testErrorRateEachRepetition[numTraining], testRecallEachRepetition[numTraining], testSpecificityEachRepetition[numTraining], testPrecisionEachRepetition[numTraining], testNPVEachRepetition[numTraining], testF1EachRepetition[numTraining], testConfusionMatrixEachRepetition[:,:,numTraining]) =
                confusionMatrix(collect(ann(Float32.(testInputs'))'), testTargets);

            @assert( isapprox( testAccuracyEachRepetition[numTraining], sum([testConfusionMatrixEachRepetition[numClass,numClass,numTraining] for numClass in 1:length(classes)])/sum(testConfusionMatrixEachRepetition[:,:,numTraining]) ) );

        end;

        # Almacenamos las metricas como una media de las obtenidas en los entrenamientos de este fold
        testAccuracy[numFold]    = mean(testAccuracyEachRepetition);
        testErrorRate[numFold]   = mean(testErrorRateEachRepetition);
        testRecall[numFold]      = mean(testRecallEachRepetition);
        testSpecificity[numFold] = mean(testSpecificityEachRepetition);
        testPrecision[numFold]   = mean(testPrecisionEachRepetition);
        testNPV[numFold]         = mean(testNPVEachRepetition);
        testF1[numFold]          = mean(testF1EachRepetition);
        testConfusionMatrix    .+= mean(testConfusionMatrixEachRepetition, dims=3)[:,:,1];

    end; # for numFold in 1:numFolds

    return (mean(testAccuracy), std(testAccuracy)), (mean(testErrorRate), std(testErrorRate)), (mean(testRecall), std(testRecall)), (mean(testSpecificity), std(testSpecificity)), (mean(testPrecision), std(testPrecision)), (mean(testNPV), std(testNPV)), (mean(testF1), std(testF1)), testConfusionMatrix;

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
    # Extraer datos del dataset
    (inputs, targets) = dataset;

    @assert(size(inputs,1) == length(targets));
    
    # Si el modelo es una Red Neuronal Artificial, llamar a la función correspondiente
    if (modelType == :ANN)
        if haskey(modelHyperparameters, "topology")
            
            return ANNCrossValidation(modelHyperparameters["topology"], 
                dataset, crossValidationIndices;
                numExecutions = haskey(modelHyperparameters, "numExecutions") ? modelHyperparameters["numExecutions"] : 50,
                transferFunctions = haskey(modelHyperparameters, "transferFunctions") ? modelHyperparameters["transferFunction"] :  fill(σ, length(modelHyperparameters["topology"])),
                maxEpochs = haskey(modelHyperparameters,"maxEpochs") ? modelHyperparameters["maxEpochs"] : 1000,
                minLoss = haskey(modelHyperparameters,"minLoss") ? modelHyperparameters["minLoss"] : 0.0,
                learningRate = haskey(modelHyperparameters, "learningRate") ? modelHyperparameters["learningRate"] : 0.01,
                validationRatio = haskey(modelHyperparameters,"learningRate") ? modelHyperparameters["validationRatio"] : 0,
                maxEpochsVal = haskey(modelHyperparameters, "maxEpochsVal") ? modelHyperparameters["maxEpochsVal"] : 20);
        end;
    end;

    # Vectores para almacenar métricas de cada fold
    numFolds = maximum(crossValidationIndices);
    accuracy = Array{Float64,1}(undef, numFolds);
    error_rate = Array{Float64,1}(undef, numFolds);
    sensitivity = Array{Float64,1}(undef, numFolds);
    specificity = Array{Float64,1}(undef, numFolds);
    ppv = Array{Float64,1}(undef, numFolds);
    npv = Array{Float64,1}(undef, numFolds);
    f1 = Array{Float64,1}(undef, numFolds);

    # Convertir el vector de salidas a strings
    targets = string.(targets);
    
    # Calcular las clases únicas
    classes = unique(targets);
    
    # Inicializar la matriz de confusión
    confusion_matrix = zeros(Int64, length(classes), length(classes));
    
    # Para cada fold en la validación cruzada
    for numFold in 1:numFolds        
        # Extraer datos de entrenamiento y test
        X_train = inputs[crossValidationIndices.!=numFold, :];
        y_train = targets[crossValidationIndices.!=numFold];
        X_test = inputs[crossValidationIndices.==numFold,:];
        y_test = targets[crossValidationIndices.==numFold];
        
        if modelType == :DoME
            # Para DoME, usamos la función trainClassDoME
            testOutputs = trainClassDoME((X_train, y_train), X_test, modelHyperparameters["maximumNodes"]);
        
        else
            if modelType == :SVC
                @assert((modelHyperparameters["kernel"] == "linear") || (modelHyperparameters["kernel"] == "poly") || (modelHyperparameters["kernel"] == "rbf") || (modelHyperparameters["kernel"] == "sigmoid"));

                model = SVMClassifier(
                    kernel = 
                        modelHyperparameters["kernel"]=="linear"  ? LIBSVM.Kernel.Linear :
                        modelHyperparameters["kernel"]=="rbf"     ? LIBSVM.Kernel.RadialBasis :
                        modelHyperparameters["kernel"]=="poly"    ? LIBSVM.Kernel.Polynomial :
                        modelHyperparameters["kernel"]=="sigmoid" ? LIBSVM.Kernel.Sigmoid : nothing,
                    cost = Float64(modelHyperparameters["C"]),
                    gamma = Float64(get(modelHyperparameters, "gamma",  -1)),
                    degree = Int32(get(modelHyperparameters, "degree", -1)),
                    coef0 = Float64(get(modelHyperparameters, "coef0",  -1)));

            elseif modelType==:DecisionTreeClassifier
                model = DTClassifier(max_depth = modelHyperparameters["max_depth"], rng=Random.MersenneTwister(1));
            elseif modelType==:KNeighborsClassifier
                model = kNNClassifier(K = modelHyperparameters["n_neighbors"]);
            else
                error(string("Unknown model ", modelType));
            end;

            # Creamos el objeto de tipo Machine
            mach = machine(model, MLJ.table(X_train), categorical(y_train));

            # Entrenamos el modelo con el conjunto de entrenamiento
            MLJ.fit!(mach, verbosity=0)

            # Pasamos el conjunto de test
            testOutputs = MLJ.predict(mach, MLJ.table(X_test))
            # if modelType==:DecisionTreeClassifier || modelType==:KNeighborsClassifier
            if modelType!=:SVC
                testOutputs = mode.(testOutputs)
            end;
            # testOutputs = string.(testOutputs);

        end;

        # Calculamos las metricas y las almacenamos en las posiciones de este fold de cada vector
        (accuracy[numFold], error_rate[numFold], sensitivity[numFold], specificity[numFold], ppv[numFold], npv[numFold], f1[numFold], this_fold_confusion_matrix) =
            confusionMatrix(testOutputs, y_test, classes);

        @assert( isapprox( accuracy[numFold], sum([this_fold_confusion_matrix[numClass,numClass] for numClass in 1:length(classes)])/sum(this_fold_confusion_matrix) ) );

        confusion_matrix .+= this_fold_confusion_matrix;

    end; # for numFold in 1:numFolds

    return (mean(accuracy), std(accuracy)), (mean(error_rate), std(error_rate)), (mean(sensitivity), std(sensitivity)), (mean(specificity), std(specificity)), (mean(ppv), std(ppv)), (mean(npv), std(npv)), (mean(f1), std(f1)), confusion_matrix;

end;
