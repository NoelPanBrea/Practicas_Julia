
# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------
using DelimitedFiles
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
    #=
    Recibe una matriz y devuelve una tupla con una matriz con una fila, 
    con los mínimos y máximos de cada columna. (Sacado del PDF)
    =#
    min_col =  minimum(dataset, dims = 1);
    max_col = maximum(dataset, dims = 1);
    return (min_col, max_col);
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    #=
    Recibe una matriz y devuelve una tupla con una matriz con una fila, 
    con las medias y desviaciones típicas de cada columna. (Sacado del PDF)
    =#
    mean_col = mean(dataset, dims = 1);
    deviation_col = std(dataset, dims = 1);
    return (mean_col, deviation_col);
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    #=
    Recibe una matriz a normalizar y los parámetros de normalización
    Devuelve la misma matriz normalizada entre máximo y mínimo
    =#
    min_values, max_values = normalizationParameters;
    dataset .-= min_values;
    range_values = max_values .- min_values;

    dataset ./= (range_values);
    dataset[:, vec(min_values .== max_values)] .= 0;
    return dataset;
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    #=
    Recibe la matriz de datos y calcula los parámetros de normalización
    y llama a notmalizeMinMax! (que modifica la matriz)
    =#
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset);
    return normalizeMinMax!(dataset, normalizationParameters);
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    #=
    Recibe una matriz a normalizar y los parámetros de normalización
    Devuelve una nueva matriz normalizada entre máximo y mínimo
    =#
    new_dataset = copy(dataset);
    min_values, max_values = normalizationParameters;
    new_dataset .-= min_values;
    range_values = max_values .- min_values;

    new_dataset ./= (range_values);
    new_dataset[:, vec(min_values .== max_values)] .= 0;
    return new_dataset;
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    #=
    Recibe la matriz de datos y calcula los parámetros de normalización
    y llama a notmalizeMinMax (que NO modifica la matriz)
    =#
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset);
    normalizeMinMax(dataset, normalizationParameters);
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    #=
    Recibe una matriz a normalizar y los parámetros de normalización
    Devuelve la misma matriz normalizada entre media y desviación típica
    =#
    mean_values, desviation_values = normalizationParameters;
    dataset .-= mean_values;
    dataset ./= desviation_values;
    dataset[:, vec(desviation_values .== 0)] .= 0;
    return dataset;
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    #=
    Recibe la matriz de datos y calcula los parámetros de normalización
    y llama a notmalizeZeroMean! (que modifica la matriz)
    =#
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset);
    return normalizeZeroMean!(dataset, normalizationParameters);
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    #=
    Recibe una matriz a normalizar y los parámetros de normalización
    Devuelve una nueva matriz normalizada entre máximo y mínimo
    =#
    new_dataset = copy(dataset);
    mean_values, desviation_values = normalizationParameters;

    new_dataset .-= mean_values;
    new_dataset ./= desviation_values;
    dataset[:, vec(desviation_values .== 0)] .= 0;
    return dataset;

end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    #=
    Recibe la matriz de datos y calcula los parámetros de normalización
    y llama a notmalizeZeroMean (que NO modifica la matriz)
    =#
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset);
    return normalizeZeroMean(dataset, normalizationParameters);
end;

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    #=
    Clasifica el vector de outputs devolviendo un vector de valores binarios
    correspondientes
    =#
    return outputs .>= threshold;
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    #=
    Recibe una matriz y la convierte  a una matriz de valores booleanos que cada
    fila sólo tenga un valor a true, que indica la clase a la que se clasifica ese 
    patrón
    =#
    if size(outputs, 2) == 1
        #Si tiene solo una columna
        outputs = classifyOutputs(outputs[:]; threshold);
        outputs = reshape(outputs, :, 1);
        return outputs;
    else
        #Si tiene MÁS de una columna la matriz
        (_, indicesMaxEachInstance) = findmax(outputs, dims = 2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true; 
        return outputs;
    end;

end;

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #=
     Valor promedio de la comparación de ambos vectores
    =#
    @assert size(targets) == size(outputs)
    return mean(targets .== outputs);
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    #=
     Solo una columna: llamada a funcion anterior tomando como vectores
     primera columna de targets
     
     Columnas mayor que 2: mirar en qué filas no coinciden los valores 
    
    Si tiene 2 clases funciona como más de dos clases
    =#
    @assert size(targets) == size(outputs)

    if size(outputs, 2) == 1
        return mean(targets[:, 1] .== outputs[:, 1]);
    else
        classComparison = targets .== outputs;
        correctClassifications = all(classComparison, dims = 2);
        accuracy = mean(correctClassifications);
        
        return accuracy;
        #=
        2º FORMA DE CALCULARLO
        classComparison = targets .!= outputsize
        incorrectClassifications = any(classComparison, dims = 2)
        accuracy = 1 - mean(incorrectClassifications)
        =#
    end;
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #=
     Pasar el umbral al vector outputs y llamar a la función anterior accuracy
    =#
    @assert length(targets) == length(outputs)
    outputs = classifyOutputs(outputs; threshold);
    return accuracy(outputs,targets);
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    #=
     1 columna: accuracy(vectores de primera columna outputs, targets)
    
     + 2 columnas: classifyOutputs(outputs) y luego accuracy()
    =#
    @assert size(targets)== size(outputs)
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
    #=
     Está mal, modificado en el ejercicio 3 IGNORAR
    =#

    dataset = reshape((inputs,targets), :, 1);
    trainClassANN(topology,dataset,transferFunctions = transferFunctions,maxEpochs = maxEpochs,minLoss = minLoss,learningRate = learningRate);
end;

dataset = readdlm("Practica2/optical+recognition+of+handwritten+digits/optdigits.full",',');
datatest = readdlm("Practica2/optical+recognition+of+handwritten+digits/optdigits.tes", ',')
begin
    inputs = dataset[:,1:64];
    test_inputs = datatest[:,1:64]
    test_inputs = Float32.(test_inputs);
    test_targets = datatest[:,65]
    test_targets = oneHotEncoding(test_targets) 
    # Con cualquiera de estas 3 maneras podemos convertir la matriz de entradas de tipo Array{Any,2} en Array{Float32,2}, si los valores son numéricos:
    inputs = Float32.(inputs);
    inputs = convert(Array{Float32,2},inputs);
    inputs = [Float32(x) for x in inputs];
    # inputs = normalizeMinMax!(inputs);
    # test_inputs = normalizeMinMax!(test_inputs);
    println("Tamaño de la matriz de entradas: ", size(inputs,1), "x", size(inputs,2), " de tipo ", typeof(inputs));
    targets = dataset[:,65];
    println("Longitud del vector de salidas deseadas antes de codificar: ", length(targets), " de tipo ", typeof(targets));
    targets = oneHotEncoding(targets);
    println("Tamaño de la matriz de salidas deseadas despues de codificar: ", size(targets,1), "x", size(targets,2), " de tipo ", typeof(targets));
    ann, losses = trainClassANN([15, 15], (test_inputs, test_targets); maxEpochs = 500, learningRate = 0.01);
    println(length(losses));
    println(accuracy(ann(permutedims(inputs))', targets));
    println(accuracy(ann(permutedims(test_inputs))', test_targets));
    # print(classifyOutputs(ann([0,1,6,15,12,1,0,0,0,7,16,6,6,10,0,0,0,8,16,2,0,11,2,0,0,5,16,3,0,5,7,0,0,7,13,3,0,8,7,0,0,4,12,0,1,13,5,0,0,0,14,9,15,9,0,0,0,0,6,14,7,1,0,0])));
end;