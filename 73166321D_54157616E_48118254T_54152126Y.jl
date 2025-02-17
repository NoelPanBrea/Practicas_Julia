
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
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
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



