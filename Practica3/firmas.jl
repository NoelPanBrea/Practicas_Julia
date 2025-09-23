

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import FileIO.load
using DelimitedFiles
using JLD2
using Images

function fileNamesFolder(folderName::String, extension::String)
    extension = uppercase(extension);
    fileNames = sort(filter(f -> endswith(uppercase(f), ".$extension"), readdir(folderName)));
    return convert(Vector{String}, first.(split.(fileNames, ".")));
end;


function loadDataset(datasetName::String, datasetFolder::String;
    datasetType::DataType=Float32)
    try
        dataset = readdlm(joinpath(datasetFolder, join([datasetName, ".tsv"])), '\t');
        target_column_index = findfirst(isequal("target"), dataset[1, 1:end]);
        inputs = convert(Matrix{datasetType}, dataset[2:end, 1:end.!=target_column_index]);
        targets = dataset[2:end, target_column_index];
        classes = sort(unique(targets));
        targets = convert(Array{Bool, 1}, targets .== classes[1]);
        return (inputs, targets);
    catch error
        print("Error: $error");
        return nothing;
    end;
end;


function loadImage(imageName::String, datasetFolder::String;
    datasetType::DataType=Float32, resolution::Int=128)
    try
        imagePath = joinpath(datasetFolder, join([imageName, ".tif"]));
        image = load(imagePath);
        resized_image = imresize(image, (resolution, resolution));
        return convert(Array{datasetType}, gray.(resized_image));
    catch error
        print("Error: $error")
        return nothing
    end;
end;


function convertImagesNCHW(imageVector::Vector{<:AbstractArray{<:Real,2}})
    imagesNCHW = Array{eltype(imageVector[1]), 4}(undef, length(imageVector), 1, size(imageVector[1],1), size(imageVector[1],2));
    for numImage in Base.OneTo(length(imageVector))
        imagesNCHW[numImage,1,:,:] .= imageVector[numImage];
    end;
    return imagesNCHW;
end;


function loadImagesNCHW(datasetFolder::String;
    datasetType::DataType=Float32, resolution::Int=128)
   imageNames = fileNamesFolder(datasetFolder, "tif");
   print(imageNames)
   convertImagesNCHW(loadImage.(imageNames, datasetFolder, datasetType = datasetType, resolution = resolution));
end;


showImage(image      ::AbstractArray{<:Real,2}                                      ) = display(Gray.(image));
showImage(imagesNCHW ::AbstractArray{<:Real,4}                                      ) = display(Gray.(     hcat([imagesNCHW[ i,1,:,:] for i in 1:size(imagesNCHW ,1)]...)));
showImage(imagesNCHW1::AbstractArray{<:Real,4}, imagesNCHW2::AbstractArray{<:Real,4}) = display(Gray.(vcat(hcat([imagesNCHW1[i,1,:,:] for i in 1:size(imagesNCHW1,1)]...), hcat([imagesNCHW2[i,1,:,:] for i in 1:size(imagesNCHW2,1)]...))));



function loadMNISTDataset(datasetFolder::String; labels::AbstractArray{Int,1}=0:9, datasetType::DataType=Float32)
    dataset = JLD2.load(joinpath(datasetFolder, "MNIST.jld2"));
    test_targets, train_targets = dataset["test_labels"], dataset["train_labels"];
    train_images, test_images = dataset["train_imgs"], dataset["test_imgs"];
    if  in(-1, labels)
        train_targets[.!in.(train_targets, [setdiff(labels,-1)])] .= -1; 
        test_targets[.!in.(test_targets, [setdiff(labels,-1)])] .= -1;
    else
        train_images = train_images[in.(train_targets, [labels])];
        test_images = test_images[in.(test_targets, [labels])];
        train_targets = train_targets[in.(train_targets, [labels])];
        test_targets = test_targets[in.(test_targets, [labels])];
    end;
    return (convert(Array{datasetType}, convertImagesNCHW(train_images)), train_targets, convert(Array{datasetType}, convertImagesNCHW(test_images)), test_targets)

end;


function intervalDiscreteVector(data::AbstractArray{<:Real,1})
    # Ordenar los datos
    uniqueData = sort(unique(data));
    # Obtener diferencias entre elementos consecutivos
    differences = sort(diff(uniqueData));
    # Tomar la diferencia menor
    minDifference = differences[1];
    # Si todas las diferencias son multiplos exactos (valores enteros) de esa diferencia, entonces es un vector de valores discretos
    isInteger(x::Float64, tol::Float64) = abs(round(x)-x) < tol
    return all(isInteger.(differences./minDifference, 1e-3)) ? minDifference : 0.
end


function cyclicalEncoding(data::AbstractArray{<:Real,1})
    # 1) vector vacío → devolvemos vectores vacíos
    isempty(data) && return (Float64[], Float64[])

    # 2) si todos los valores son iguales, evitamos llamar a intervalDiscreteVector
    xmin, xmax = extrema(data)
    if xmin == xmax
        n = length(data)
        return (zeros(n), ones(n))   # ángulo 0 → sin=0, cos=1
    end

    # 3) caso normal: usamos intervalDiscreteVector como pide el enunciado
    m = intervalDiscreteVector(data)   # 0 si continuo; >0 si discreto
    denom = (xmax - xmin) + m          # siempre >0 aquí porque xmin!=xmax

    angles = ((data .- xmin) ./ denom) .* (2pi)
    return (sin.(angles), cos.(angles))
end;


function loadStreamLearningDataset(datasetFolder::String; datasetType::DataType=Float32)
    fX = abspath(joinpath(datasetFolder, "elec2_data.dat"))
    fy = abspath(joinpath(datasetFolder, "elec2_label.dat"))
    (isfile(fX) && isfile(fy)) || return nothing

    # ── leer X con autodetección de separador (tab -> espacio -> coma)
    Xraw = readdlm(fX, '\t', Any; comments=false)
    if size(Xraw,2) == 1
        Xraw = readdlm(fX, ' ', Any; comments=false)
        if size(Xraw,2) == 1
            Xraw = readdlm(fX, ',', Any; comments=false)
        end
    end

    # ── leer y con la misma lógica
    yraw = readdlm(fy, '\t', Any; comments=false)
    if size(yraw,2) > 1
        # si viniera en varias columnas, lo aplanamos
        yraw = vec(yraw[:,1])
    else
        yraw = vec(yraw)
        if length(yraw) == 1
            # por si quedó todo en una “celda” gigante (raro), último intento:
            yraw = vec(readdlm(fy, ' ', Any; comments=false))
            if length(yraw) == 1
                yraw = vec(readdlm(fy, ',', Any; comments=false))
            end
        end
    end

    # Esperamos 8 columnas: date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer
    if size(Xraw,2) < 8
        error("Elec2: esperaba 8 columnas en elec2_data.dat, obtuve $(size(Xraw)). Revisa el separador/archivo.")
    end

    # Quitamos 1 (date) y 4 (nswprice) → quedan 6: day, period, nswdemand, vicprice, vicdemand, transfer
    keep = setdiff(1:size(Xraw,2), [1,4])
    Xkeep_any = Xraw[:, keep]

    # day → sin/cos con cyclicalEncoding
    toFloat(x) = x isa Number ? float(x) : parse(Float64, string(x))
    day_real = toFloat.(Xkeep_any[:, 1])
    s, c = cyclicalEncoding(day_real)

    # Resto: period, nswdemand, vicprice, vicdemand, transfer
    Xrest = convert.(datasetType, toFloat.(Xkeep_any[:, 2:end]))

    # Ensamblar en orden requerido: sin, cos, period, nswdemand, vicprice, vicdemand, transfer
    X = hcat(convert.(datasetType, s),
             convert.(datasetType, c),
             Xrest)

    # Etiquetas Bool (acepta 1/0 y UP/DOWN, etc.)
    toBool(x) = x isa Number ? (x == 1) :
                (lowercase(strip(string(x))) in ("1","true","t","yes","y","up"))
    y = map(toBool, yraw)

    return (X, y)
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Flux

indexOutputLayer(ann::Chain) = length(ann) - (ann[end]==softmax);

function newClassCascadeNetwork(numInputs::Int, numOutputs::Int)
    ann = Chain();
    if numOutputs > 2
        ann = Chain(ann..., Dense(numInputs, numOutputs, identity), softmax);
    else
        ann = Chain(ann..., Dense(numInputs, 1, σ));
    end;
    return ann;
end;

function addClassCascadeNeuron(previousANN::Chain; transferFunction::Function=σ)
    outputLayer = previousANN[indexOutputLayer(previousANN)];
    previousLayers = previousANN[1:(indexOutputLayer(previousANN)-1)];
    numInputsOutputLayer  = size(outputLayer.weight, 2);
    numOutputsOutputLayer = size(outputLayer.weight, 1);
    weight = outputLayer.weight;
    bias = outputLayer.bias;
    if numOutputsOutputLayer > 2
        ann = Chain(previousLayers..., SkipConnection(Dense(numInputsOutputLayer, 1, transferFunction), (mx, x) -> vcat(x, mx)), Dense(numInputsOutputLayer + 1, numOutputsOutputLayer, identity), softmax);
    else
        ann = Chain(previousLayers..., SkipConnection(Dense(numInputsOutputLayer, 1, transferFunction), (mx, x) -> vcat(x, mx)), Dense(numInputsOutputLayer + 1, 1, σ));
    end;
    ann[indexOutputLayer(ann)].weight[:, 1:end - 1] = weight;
    ann[indexOutputLayer(ann)].weight[:, end] .= 0;
    ann[indexOutputLayer(ann)].bias .= bias;

    return ann;
end;

function trainClassANN!(ann::Chain, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, trainOnly2LastLayers::Bool;
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)

    # prepara el dataset
    inputs = convert(AbstractArray{Float32,2}, trainingDataset[1]);
    targets = trainingDataset[2];

    # función loss dependiendo de si es salida binaria o multiclase
    loss(model, x, y) = (size(y, 1) == 1) ?
        Losses.binarycrossentropy(model(x), y) :
        Losses.crossentropy(model(x), y);

    # inicializa función loss en la iteracion 0 del bucle de entrenamiento
    trainingLosses = Float32[loss(ann, inputs, targets)];

    # optimizador
    opt_state = Flux.setup(Adam(learningRate), ann);

    # congela las dos últimas capas si es necesario
    if trainOnly2LastLayers
        Flux.freeze!(opt_state.layers[1:(length(ann)-2)]);
    end;

    # bucle de entrenamiento
    for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(inputs, targets)], opt_state);
        current_loss = Float32(loss(ann, inputs, targets));
        push!(trainingLosses, current_loss);

        # primer criterio de parada
        if current_loss <= minLoss
            break;
        end;

        # segundo criterio de parada
        if length(trainingLosses) >= lossChangeWindowSize + 1
            lossWindow = trainingLosses[end - lossChangeWindowSize + 1:end];
            minLossValue, maxLossValue = extrema(lossWindow);
            if minLossValue > 0
                if ((maxLossValue - minLossValue) / minLossValue) <= minLossChange
                    break;
                end;
            end;
        end;
    end;

    return trainingLosses
end;


function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunction::Function=σ,
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    #
    # Codigo a desarrollar
    #
end;

function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunction::Function=σ,
    maxEpochs::Int=100, minLoss::Real=0.0, learningRate::Real=0.01, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    #
    # Codigo a desarrollar
    #
end;
    

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

HopfieldNet = Array{Float32,2}

function trainHopfield(trainingSet::AbstractArray{<:Real,2})
    #
    # Codigo a desarrollar
    #
end;
function trainHopfield(trainingSet::AbstractArray{<:Bool,2})
    #
    # Codigo a desarrollar
    #
end;
function trainHopfield(trainingSetNCHW::AbstractArray{<:Bool,4})
    #
    # Codigo a desarrollar
    #
end;

function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;
function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Bool,1})
    #
    # Codigo a desarrollar
    #
end;


function runHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    prev_S = nothing;
    prev_prev_S = nothing;
    while S!=prev_S && S!=prev_prev_S
        prev_prev_S = prev_S;
        prev_S = S;
        S = stepHopfield(ann, S);
    end;
    return S
end;
function runHopfield(ann::HopfieldNet, dataset::AbstractArray{<:Real,2})
    outputs = copy(dataset);
    for i in 1:size(dataset,1)
        outputs[i,:] .= runHopfield(ann, view(dataset, i, :));
    end;
    return outputs;
end;
function runHopfield(ann::HopfieldNet, datasetNCHW::AbstractArray{<:Real,4})
    outputs = runHopfield(ann, reshape(datasetNCHW, size(datasetNCHW,1), size(datasetNCHW,3)*size(datasetNCHW,4)));
    return reshape(outputs, size(datasetNCHW,1), 1, size(datasetNCHW,3), size(datasetNCHW,4));
end;





function addNoise(datasetNCHW::AbstractArray{<:Bool,4}, ratioNoise::Real)
    #
    # Codigo a desarrollar
    #
end;

function cropImages(datasetNCHW::AbstractArray{<:Bool,4}, ratioCrop::Real)
    #
    # Codigo a desarrollar
    #
end;

function randomImages(numImages::Int, resolution::Int)
    #
    # Codigo a desarrollar
    #
end;

function averageMNISTImages(imageArray::AbstractArray{<:Real,4}, labelArray::AbstractArray{Int,1})
    #
    # Codigo a desarrollar
    #
end;

function classifyMNISTImages(imageArray::AbstractArray{<:Bool,4}, templateInputs::AbstractArray{<:Bool,4}, templateLabels::AbstractArray{Int,1})
    #
    # Codigo a desarrollar
    #
end;

function calculateMNISTAccuracies(datasetFolder::String, labels::AbstractArray{Int,1}, threshold::Real)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# using ScikitLearn: @sk_import, fit!, predict
# @sk_import svm: SVC

using MLJ, LIBSVM, MLJLIBSVMInterface
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
import Main.predict
predict(model, inputs::AbstractArray) = MLJ.predict(model, MLJ.table(inputs));



using Base.Iterators
using StatsBase

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function batchInputs(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchTargets(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function batchLength(batch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function selectInstances(batch::Batch, indices::Any)
    #
    # Codigo a desarrollar
    #
end;

function joinBatches(batch1::Batch, batch2::Batch)
    #
    # Codigo a desarrollar
    #
end;


function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )
    #
    # Codigo a desarrollar
    #
end;

function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
    #
    # Codigo a desarrollar
    #
end;

function addBatch!(memory::Batch, newBatch::Batch)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    #
    # Codigo a desarrollar
    #
end;

function euclideanDistances(dataset::Batch, instance::AbstractArray{<:Real,1})
    #
    # Codigo a desarrollar
    #
end;

function nearestElements(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    #
    # Codigo a desarrollar
    #
end;

function streamLearning_KNN(datasetFolder::String, windowSize::Int, batchSize::Int, k::Int)
    #
    # Codigo a desarrollar
    #
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function predictKNN_SVM(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int, C::Real)
    #
    # Codigo a desarrollar
    #
end;

function predictKNN_SVM(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int, C::Real)
    #
    # Codigo a desarrollar
    #
end;

