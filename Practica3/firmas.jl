

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import FileIO.load
using DelimitedFiles
using JLD2
using Images

#No eliminar
# function fileNamesFolder(folderName::String, extension::String)
#     extension = uppercase(extension);
#     fileNames = sort(filter(f -> endswith(uppercase(f), ".$extension"), readdir(folderName)));
#     return convert(Vector{String}, first.(split.(fileNames, ".")));
# end;


# function loadDataset(datasetName::String, datasetFolder::String;
#     datasetType::DataType=Float32)
#     try
#         dataset = readdlm(joinpath(datasetFolder, join([datasetName, ".tsv"])), '\t');
#         target_column_index = findfirst(isequal("target"), dataset[1, 1:end]);
#         inputs = convert(Matrix{datasetType}, dataset[2:end, 1:end.!=target_column_index]);
#         targets = dataset[2:end, target_column_index];
#         classes = sort(unique(targets));
#         targets = convert(Array{Bool, 1}, targets .== classes[1]);
#         return (inputs, targets);
#     catch error
#         print("Error: $error");
#         return nothing;
#     end;
# end;

function fileNamesFolder(folderName::String, extension::String)
    isdir(folderName) || return String[]
    extU = uppercase(extension)
    files = sort(filter(f -> endswith(uppercase(f), ".$extU"), readdir(folderName)))
    return map(f -> first(splitext(f)), files)
end


function loadDataset(datasetName::String, datasetFolder::String;
    datasetType::DataType=Float32)
    fname = datasetName * ".tsv"
    fpath = abspath(joinpath(datasetFolder, fname))
    isfile(fpath) || return nothing

    M = readdlm(fpath, '\t', Any)

    headers = String.(M[1, :])      # cabeceras
    data    = M[2:end, :]           # datos

    tgt_idx = findfirst(==("target"), headers)
    tgt_idx === nothing && return nothing

    incols = setdiff(1:size(M,2), [tgt_idx])
    X_any  = data[:, incols]
    y_any  = vec(data[:, tgt_idx])

    # convertir entradas a floats
    toFloat(x) = x isa Number ? float(x) : parse(Float64, string(x))
    X = convert.(datasetType, toFloat.(X_any))

    # convertir target a Bool (soporta 0/1 y varios textos)
    toBool(x) = x isa Number ? (x == 1) :
                (lowercase(strip(string(x))) in ("1","true","t","yes","y","up"))
    y = map(toBool, y_any)

    return (X, y)
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
    newLayer = SkipConnection(Dense(numInputsOutputLayer, 1, transferFunction), (mx, x) -> vcat(x, mx))
    if numOutputsOutputLayer > 2
        ann = Chain(previousLayers..., newLayer, Dense(numInputsOutputLayer + 1, numOutputsOutputLayer, identity), softmax);
    else
        ann = Chain(previousLayers..., newLayer, Dense(numInputsOutputLayer + 1, 1, σ));
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
        Flux.Losses.binarycrossentropy(model(x), y) :
        Flux.Losses.crossentropy(model(x), y);

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
        if length(trainingLosses) >= lossChangeWindowSize
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
    
    
    inputs, outputs = trainingDataset
    inputs = convert(Array{Float32}, inputs')
    outputs = convert(Array{Bool}, outputs') 

    ann = newClassCascadeNetwork(size(inputs, 1), size(outputs, 1))

    loss = trainClassANN!(ann, (inputs, outputs),
        false;  
        maxEpochs = maxEpochs, 
        minLoss = minLoss, 
        learningRate = learningRate, 
        minLossChange = minLossChange, 
        lossChangeWindowSize = lossChangeWindowSize)
    
    #añadimos neuronas una a una
    for i in 1:maxNumNeurons
        ann = addClassCascadeNeuron(ann; transferFunction = transferFunction)
        
        #entrenar con conexiones congeladas
        if i > 1
            loss_2 = trainClassANN!(ann, (inputs, outputs),
                true;
                maxEpochs = maxEpochs, 
                minLoss = minLoss, 
                learningRate = learningRate, 
                minLossChange = minLossChange, 
                lossChangeWindowSize = lossChangeWindowSize)

            loss = append!(loss, loss_2[2:end])
        end

        #entrenar la red entera
        loss_2 = trainClassANN!(ann, (inputs, outputs),
            false;
            maxEpochs = maxEpochs, 
            minLoss = minLoss, 
            learningRate = learningRate, 
            minLossChange = minLossChange, 
            lossChangeWindowSize = lossChangeWindowSize)

        loss = append!(loss, loss_2[2:end])
    end
    return ann, loss
end;

function trainClassCascadeANN(maxNumNeurons::Int,
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunction::Function=σ,
    maxEpochs::Int=100, minLoss::Real=0.0, learningRate::Real=0.01, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    
    #Función análoga a la anterior pero en lugar de recibir una matriz de salidas, recibe un vector de salidas
    inputs, outputs_vec = trainingDataset
    outputs_mat = reshape(outputs_vec, :, 1)

    return trainClassCascadeANN(maxNumNeurons, (inputs, outputs_mat); 
        transferFunction = transferFunction, 
        maxEpochs = maxEpochs, 
        minLoss = minLoss, 
        learningRate = learningRate, 
        minLossChange = minLossChange, 
        lossChangeWindowSize = lossChangeWindowSize)
end;
    

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

HopfieldNet = Array{Float32,2}

function trainHopfield(trainingSet::AbstractArray{<:Real,2})
    N, m = size(trainingSet)
    W = (trainingSet' * trainingSet) ./ N          
    # poner diagonal a 0
    for i in 1:m
        W[i,i] = 0
    end
    return HopfieldNet(W)
end;
function trainHopfield(trainingSet::AbstractArray{<:Bool,2})
    S = (2 .* trainingSet) .- 1
    return trainHopfield(S)
end;
function trainHopfield(trainingSetNCHW::AbstractArray{<:Bool,4})
    N = size(trainingSetNCHW, 1) #numero de patrones
    m = prod(size(trainingSetNCHW)) ÷ N
    S = reshape(trainingSetNCHW, N, m) #reordenar a matriz Nxm
    return trainHopfield(S)
end;

function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Real,1})
    x = Float32.(S)
    y = ann * x #calcular activación
    return sign.(y) .|> Float32 #aplicamos signo
end;
function stepHopfield(ann::HopfieldNet, S::AbstractArray{<:Bool,1})
    S_real = (2 .* S) .- 1
    y = stepHopfield(ann, S_real)
    return y .>= 0 #devolvemos como booleano
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
    noiseSet = copy(datasetNCHW)
    numPixels = length(noiseSet) #total valores
    numNoisy  = Int(round(numPixels * ratioNoise)) #cuantos invertimos
    if numNoisy > 0
        indices = randperm(numPixels)[1:numNoisy]
        noiseSet[indices] .= .!noiseSet[indices]  # invertir valores
    end
    return noiseSet
end;

function cropImages(datasetNCHW::AbstractArray{<:Bool,4}, ratioCrop::Real)
    croppedSet = copy(datasetNCHW)
    N, C, H, W = size(croppedSet) #dimension dataset
    colsToRemove = ceil(Int, W * ratioCrop) #columnas a eliminar
    if colsToRemove > 0
        croppedSet[:,:,:,end-colsToRemove+1:end] .= false
    end
    return croppedSet
end;

function randomImages(numImages::Int, resolution::Int)
    return randn(numImages, 1, resolution, resolution) .>= 0
end;

function averageMNISTImages(imageArray::AbstractArray{<:Real,4}, labelArray::AbstractArray{Int,1})
    labels = unique(labelArray)
    N = length(labels)
    C, H, W = size(imageArray, 2), size(imageArray, 3), size(imageArray, 4)

    templateArray = similar(imageArray, eltype(imageArray), (N, C, H, W)) # creamos la matriz de salida en formato NCHW

    for indexLabel in 1:N
        templateArray[indexLabel, 1, :, :] .= dropdims(mean(imageArray[labelArray.==labels[indexLabel], 1, :, :], dims=1), dims=1)
    end

    return (templateArray, labels)
end;

function classifyMNISTImages(imageArray::AbstractArray{<:Bool,4}, templateInputs::AbstractArray{<:Bool,4}, templateLabels::AbstractArray{Int,1})
    outputs = fill(-1, size(imageArray, 1))

    for i in 1:length(templateLabels)
        template = templateInputs[[i], :, :, :]
        indicesCoincidence = vec(all(imageArray .== template, dims=[3,4]))   
        outputs[indicesCoincidence] .= templateLabels[i]
    end

    return outputs
end;

function calculateMNISTAccuracies(datasetFolder::String, labels::AbstractArray{Int,1}, threshold::Real)
    dataset = loadMNISTDataset(datasetFolder, labels = labels);
    templateArray, templateLabels = averageMNISTImages(dataset[1], dataset[2]);
    trainImages = dataset[1] .>= threshold;
    testImages = dataset[3] .>= threshold;
    templateArray = templateArray .>= threshold;
    trainedNet = trainHopfield(templateArray);
    resultMatrix = runHopfield(trainedNet, trainImages);
    booltrainVector = dataset[2] .== classifyMNISTImages(resultMatrix, templateArray, templateLabels);
    resultMatrix = runHopfield(trainedNet, testImages);
    booltestVector = dataset[4] .== classifyMNISTImages(resultMatrix, templateArray, templateLabels);
    return (count(booltrainVector) / length(booltrainVector), count(booltestVector) / length(booltestVector))

end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

# using ScikitLearn: @sk_import, fit!, predict
# @sk_import svm: SVC

using MLJ, LIBSVM, MLJLIBSVMInterface
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
import Main.predict
predict(model, inputs::AbstractArray) = (outputs = MLJ.predict(model, MLJ.table(inputs)); return levels(outputs)[int(outputs)]; )


using Base.Iterators
using StatsBase

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function batchInputs(batch::Batch)
    return batch[1]
end;


function batchTargets(batch::Batch)
    return batch[2]
end;


function batchLength(batch::Batch)
    return size(batch[1], 1)
end;

function selectInstances(batch::Batch, indices::Any)
    return (batch[1][indices, :], batch[2][indices])
end;

function joinBatches(batch1::Batch, batch2::Batch)
    return (vcat(batch1[1], batch2[1]), vcat(batch1[2], batch2[2]))
end;


function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    inputs, targets = dataset
    N = size(inputs, 1)

    indices = shuffleRows ? randperm(N) : collect(1:N)
    batches = [indices[i:min(i+batchSize-1, N)] for i in 1:batchSize:N]
    return [selectInstances(dataset, b) for b in batches]
end;

function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )
    training = joinBatches(supportVectors, dataset)

    k = kernel == "linear" ? LIBSVM.Kernel.Linear :
        kernel == "rbf"   ? LIBSVM.Kernel.RadialBasis :
        kernel == "poly"  ? LIBSVM.Kernel.Polynomial :
        kernel == "sigmoid" ? LIBSVM.Kernel.Sigmoid :
        error("Unknown kernel: $kernel")

    model = SVMClassifier(kernel = k,
                          cost = Float64(C),
                          gamma = Float64(gamma),
                          degree = Int32(round(degree)),
                          coef0 = Float64(coef0))

    mach = machine(model, MLJ.table(batchInputs(training)), categorical(batchTargets(training)))
    MLJ.fit!(mach)

    indicesNewSupportVectors = sort(mach.fitresult[1].SVs.indices)

    N = batchLength(supportVectors)
    idx_old = filter(i -> i <= N, indicesNewSupportVectors)
    idx_new = filter(i -> i > N, indicesNewSupportVectors)
    idx_new = map(i -> i - N, idx_new)

    sv_from_old = isempty(idx_old) ? ( Array{eltype(supportVectors[1]),2}(undef,0,size(supportVectors[1],2)), Array{eltype(supportVectors[2]),1}(undef,0) ) :
                 selectInstances(supportVectors, idx_old)
    sv_from_new = isempty(idx_new) ? ( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)), Array{eltype(dataset[2]),1}(undef,0) ) :
                 selectInstances(dataset, idx_new)

    newSupportVectors = joinBatches(sv_from_old, sv_from_new)

    return mach, newSupportVectors, (idx_old, idx_new)
end;

function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    svmBatch = (Array{eltype(batches[1][1]),2}(undef,0,size(batches[1][1],2)), Array{eltype(batches[1][2]),1}(undef,0) )
    model = 0
    for batch in batches
        model, svmBatch, _ = trainSVM(batch, kernel, C, degree = degree, gamma = gamma, coef0 = coef0, supportVectors = svmBatch)
    end;
    return model;
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
    full_dataset = loadStreamLearningDataset(datasetFolder)
    memory = selectInstances(full_dataset, 1:windowSize)
    rest = selectInstances(full_dataset, (windowSize+1):length(full_dataset[2]))
    batches = divideBatches(rest, batchSize; shuffleRows=false)
    return memory, batches
end;

function addBatch!(memory::Batch, newBatch::Batch)
    memory_inputs, memory_targets = memory
    new_inputs, new_targets = newBatch

    len_new_instances = size(new_inputs, 1)         
    len_memory = size(memory_inputs, 1)           

    # desplaza la memoria, es decir, quita primeras instancias
    memory_inputs[1:(len_memory-len_new_instances), :] .= memory_inputs[(len_new_instances+1):len_memory, :]
    memory_targets[1:(len_memory-len_new_instances)] .= memory_targets[(len_new_instances+1):len_memory]

    # copia nuevos datos al final
    memory_inputs[(len_memory-len_new_instances+1):len_memory, :] .= new_inputs
    memory_targets[(len_memory-len_new_instances+1):len_memory] .= new_targets

    return nothing   # no devuelve nada porque solo modifica la variable memory
end;

function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)

    memory, batches = initializeStreamLearningData(datasetFolder, windowSize, batchSize)
    model = trainSVM(memory, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
    accuracies = Float64[]

    for batch in batches
        inputs_batch, targets_batch = batch

        y_predicted = MLJ.predict(model[1], inputs_batch)  # solo nos interesa el primer elemento de la tupla "model", que es mach
        accuracy = mean(y_predicted .== targets_batch)
        push!(accuracies, accuracy)

        addBatch!(memory, batch)

        model = trainSVM(memory, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
    end

    return accuracies
end;

function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)

    firstBatch, batches = initializeStreamLearningData(datasetFolder, batchSize, batchSize)
    
    if length(unique(batchTargets(firstBatch))) < 2
        error("El primer batch debe contener al menos dos clases diferentes")
    end

    model, supportVectors, (supportIndex, indicesSupportVectorsInFirstBatch) =
        trainSVM(firstBatch, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
    if model === nothing
        error("No se pudo entrenar el modelo inicial")
    end

    ageVector   = collect(batchLength(firstBatch):-1:1)
    supportAges = ageVector[indicesSupportVectorsInFirstBatch]

    accuracies = Float64[]

    for current in batches
        inputs  = batchInputs(current)
        targets = batchTargets(current)
        Nb      = batchLength(current)

        predictions = MLJ.predict(model, inputs)
        accuracy    = mean(predictions .== targets)
        push!(accuracies, accuracy)

        if !isempty(supportAges)
            supportAges .+= Nb
        end

        if !isempty(supportAges)
            selected_indices = findall(x -> x <= windowSize, supportAges)
            if isempty(selected_indices)
                supportVectors = (zeros(eltype(inputs), 0, size(inputs,2)), Int[])
                supportAges    = Int[]
            else
                supportVectors = selectInstances(supportVectors, selected_indices)
                supportAges    = supportAges[selected_indices]
            end
        end

        newModel, _, (newSupportIndex, newBatchIndex) =
            trainSVM(current, kernel, C; degree=degree, gamma=gamma, coef0=coef0,
                     supportVectors=supportVectors)

        if newModel !== nothing
            # Actualizar conjunto de SV
            prevSel   = isempty(supportVectors[2]) ? supportVectors :
                        selectInstances(supportVectors, newSupportIndex)
            batchSel  = selectInstances(current, newBatchIndex)
            supportVectors = joinBatches(prevSel, batchSel)

            prevAgesSel     = isempty(supportAges) ? Int[] : supportAges[newSupportIndex]
            newBatchAges    = collect(Nb:-1:1)
            newSupportAges  = newBatchAges[newBatchIndex]
            supportAges     = vcat(prevAgesSel, newSupportAges)

            model = newModel
        end
    end

    return accuracies
end;


function euclideanDistances(dataset::Batch, instance::AbstractArray{<:Real,1})
    inputs = batchInputs(dataset)
    diffs = inputs .- instance'      # trasponemos instance
    return sqrt.(sum(diffs.^2, dims=2))[:, 1]
end;

function nearestElements(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    distances = euclideanDistances(dataset, instance)
    idx = partialsortperm(distances, 1:k)
    return selectInstances(dataset, idx)
end;

function predictKNN(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    inputs = batchInputs(dataset)
    targets = batchTargets(dataset)
    distances = euclideanDistances(permutedims(inputs), vec(instance))  
    k_indices = partialsortperm(distances, 1:k)
    outputs = targets[k_indices]
    return mode(outputs)
end;

function predictKNN(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    return [predictKNN(dataset, instance, k) for instance in eachrow(instances)]
end;

function streamLearning_KNN(datasetFolder::String, windowSize::Int, batchSize::Int, k::Int)
    memory, batches = initializeStreamLearningData(datasetFolder, windowSize, batchSize)
    accuracies = Float64[]

    for batch in batches
        inputs_batch, targets_batch = batch
        y_predicted = predictKNN(memory, inputs_batch, k)
        accuracy = mean(y_predicted .== targets_batch)
        push!(accuracies, accuracy)
        addBatch!(memory, batch)
    end

    return accuracies
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function predictKNN_SVM(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int, C::Real)
    inputs_knn, targets_knn = nearestElements(dataset, instance, k)
    if length(unique(targets_knn)) == 1
        return targets_knn[1]
    end

    svm = SVMClassifier(kernel = "linear", cost = Float64(C))
    # hacemos permutedims para que los inputs estén en filas y vec por si los targets son una matriz de una fila
    mach = machine(svm, permutedims(inputs_knn), vec(targets_knn))
    MLJ.fit!(mach)
    prediction = predict(mach, reshape(instance, 1, :))

    return prediction[1]
end;

function predictKNN_SVM(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int, C::Real)
    return [predictKNN_SVM(dataset, instance, k, C) for instance in eachrow(instances)]
end;

