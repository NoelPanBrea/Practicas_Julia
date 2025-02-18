
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
    min_values, max_values = normalizationParameters[1], normalizationParameters[2];
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
    min_values, max_values = normalizationParameters[1], normalizationParameters[2];
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
    mean_values, desviation_values = normalizationParameters[1], normalizationParameters[2];
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
 
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
   
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
    return sum(targets .== outputs) / length(targets);
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(outputs, 2) == 1
        return accuracy(outputs[:, 1], targets[:, 1]);
    else
        classComparison = targets .== outputs;
        correctClassifications = all(classComparison, dims = 2);
        accuracy = mean(correctClassifications);
        
        return accuracy;
    end;
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    outputs = classifyOutputs(outputs; threshold);
    return accuracy(outputs,targets);
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
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
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    dataset = reshape((inputs,targets), :, 1);
    trainClassANN(topology,dataset,transferFunctions,maxEpochs,minLoss,learningRate)
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    permutation = Random.randperm(N);
    test_index = permutation[1:(int(round(N*P)))];
    train_index = permutation[(int(round(N*P))):end];
    index = (train_index, test_index);
    return index;
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    indexNoVal = holdOut(N, Ptest);
    new_N = indexNoVal[1];
    new_Pval = (N/new_N)*Pval;
    indexWithVal = holdOut(new_N, new_Pval);
    return indexWithVal;
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum(outputs .& targets);
    VN = sum(.!outputs .& .!targets);
    FP = sum(outputs .& .!targets);
    FN = sum(.!outputs .& targets);

    matriz_confusion = [VP FP; FN VN];

    precision = (VN + VP) / (VN + VP + FN + FP);
    tasa_error = (FN + FP) / (VN + VP + FN + FP);
    sensibilidad = VP / (FN + VP);
    especificidad = VN / (FP + VN);
    valor_predictivo_positivo = VP / (VP + FP);
    valor_predictivo_negativo = VN / (VN + FN);
    f1_score = (2*valor_predictivo_positivo*sensibilidad)/ (valor_predictivo_positivo + sensibilidad);

    if VP == 0 & FN == 0  
        sensibilidad = 1;
    end
    if VP == 0 & FP == 0  
        valor_predictivo_positivo = 1;
    end
    if TN == 0 & FP == 0  
        especificidad = 1;
    end
    if VN == 0 & FN == 0  
        valor_predictivo_negativo = 1  ;
    end

    if valor_predictivo_positivo == 0 & sensibilidad == 0
        f1_score = 0;

    return (precision, tasa_error, sensibilidad, especificidad, valor_predictivo_positivo, valor_predictivo_negativo, f1_score, matriz_confusion)
    

end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    new_outputs = classifyOutputs(outputs, threshold = threshold);
    confusionMatrix(new_outputs, targets);
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    
    if (size(outputs, 2) != size(targets, 2)) & size(outputs, 2) == 1
        return confusionMatrix(outputs[:,1], targets[:,1], strategy)
    end

    num_clases = size(outputs, 2)
    sensibilidad = zeros(Float64, num_classes)
    especificidad = zeros(Float64, num_classes)
    valor_predictivo_positivo = zeros(Float64, num_classes)
    valor_predictivo_negativo = zeros(Float64, num_classes)
    f1_score = zeros(Float64, num_classes)

    for i in 1:num_classes
        outputs_class = outputs[:, i]
        targets_class = targets[:, i]
        
        stats = confusionMatrix(outputs_class, targets_class)
        sensibilidad, especificidad, valor_predictivo_positivo, valor_predictivo_negativo, f1_score = stats[3:end]
    end

    matriz_confusion = [sum((outputs .== i) .& (targets .== j)) for i in 1:num_classes, j in 1:num_classes]

    instancias_clase = vec(sum(targets, dims=1))

    if weighted == true
        sensibilidad_media = sum(sensibilidad .* instancias_clase) / sum(instancias_clase)
        especificidad_media = sum(especificidad .* instancias_clase) / sum(instancias_clase)
        valor_predictivo_positivo_medio = sum(valor_predictivo_positivo .* instancias_clase) / sum(instancias_clase)
        valor_predictivo_negativo_medio = sum(valor_predictivo_negativo .* instancias_clase) / sum(instancias_clase)
        f1_score_medio = sum(f1_score .* instancias_clase) / sum(instancias_clase)

    else
        sensibilidad_media = mean(sensibilidad)
        especificidad_media = mean(especificidad)
        valor_predictivo_positivo_medio = mean(valor_predictivo_positivo)
        valor_predictivo_negativo_medio = mean(valor_predictivo_negativo)
        f1_score_medio = mean(f1_score1)
        
    end

    precision = accuracy(outputs, targets)
    tasa_error = 1 - accuracy_value

    return (precision, tasa_error, sensibilidad, especificidad, valor_predictivo_positivo, valor_predictivo_negativo, f1_score, matriz_confusion)
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
    #
    # Codigo a desarrollar
    #
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
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


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
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
    #
    # Codigo a desarrollar
    #
end;


end;
