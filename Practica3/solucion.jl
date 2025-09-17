




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import FileIO.load
using DelimitedFiles
using JLD2
using Images

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
    fpath = abspath(joinpath(datasetFolder, imageName * ".tif"))
    isfile(fpath) || return nothing

    img = load(fpath)
    imggray = Gray.(img)
    imgres  = imresize(imggray, (resolution, resolution))

    ch = channelview(imgres)
    mat = ndims(ch) == 3 && size(ch,1) == 1 ? reshape(ch, size(ch,2), size(ch,3)) : ch

    return convert.(datasetType, float.(mat))
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
    names = fileNamesFolder(datasetFolder, "tif")
    if isempty(names)
        names = fileNamesFolder(datasetFolder, "tiff")
    end

    # 2) Cargar cada imagen con loadImage (mismo resolution y tipo)
    imgs = loadImage.(names, Ref(datasetFolder);
                      datasetType=datasetType, resolution=resolution)

    # 3) Filtrar posibles `nothing` (por si algún archivo falta o falla)
    imgs_ok = filter(!isnothing, imgs)

    # 4) Si no hay ninguna válida, devolver array vacío NCHW
    if isempty(imgs_ok)
        return Array{datasetType,4}(undef, 0, 1, resolution, resolution)
    end

    # 5) Convertir a NCHW (N, C=1, H, W)
    return convertImagesNCHW(imgs_ok)
end;


showImage(image      ::AbstractArray{<:Real,2}                                      ) = display(Gray.(image));
showImage(imagesNCHW ::AbstractArray{<:Real,4}                                      ) = display(Gray.(     hcat([imagesNCHW[ i,1,:,:] for i in 1:size(imagesNCHW ,1)]...)));
showImage(imagesNCHW1::AbstractArray{<:Real,4}, imagesNCHW2::AbstractArray{<:Real,4}) = display(Gray.(vcat(hcat([imagesNCHW1[i,1,:,:] for i in 1:size(imagesNCHW1,1)]...), hcat([imagesNCHW2[i,1,:,:] for i in 1:size(imagesNCHW2,1)]...))));



function loadMNISTDataset(datasetFolder::String; labels::AbstractArray{Int,1}=0:9, datasetType::DataType=Float32)
    fpath = abspath(joinpath(datasetFolder, "MNIST.jld2"))
    isfile(fpath) || return nothing

    d = JLD2.load(fpath)  # debe contener: train_imgs, train_labels, test_imgs, test_labels
    train_imgs  = d["train_imgs"]
    train_labels = d["train_labels"]
    test_imgs   = d["test_imgs"]
    test_labels  = d["test_labels"]

    # Cast de cada imagen al tipo pedido (manteniendo [0,1])
    cast_img = img -> convert.(datasetType, img)

    # Soporte de "one-vs-rest": si labels incluye -1, reetiqueta todo lo que NO esté en (labels \ {-1}) como -1
    has_minus1 = any(==( -1), labels)
    target_set = has_minus1 ? setdiff(labels, -1) : labels

    # ── Entrenamiento
    t_lab = copy(train_labels)
    if has_minus1
        # lo que no esté en target_set pasa a -1
        t_lab .= ifelse.(in.(t_lab, Ref(target_set)), t_lab, -1)
    end
    keep_tr = in.(t_lab, Ref(labels))
    t_imgs  = train_imgs[keep_tr]
    t_y     = t_lab[keep_tr]

    # ── Test
    v_lab = copy(test_labels)
    if has_minus1
        v_lab .= ifelse.(in.(v_lab, Ref(target_set)), v_lab, -1)
    end
    keep_te = in.(v_lab, Ref(labels))
    v_imgs  = test_imgs[keep_te]
    v_y     = v_lab[keep_te]

    # A NCHW
    Xtr = convertImagesNCHW(cast_img.(t_imgs))
    Xte = convertImagesNCHW(cast_img.(v_imgs))

    return (Xtr, t_y, Xte, v_y)
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
    #
    # Codigo a desarrollar
    #
end;

function addClassCascadeNeuron(previousANN::Chain; transferFunction::Function=σ)
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN!(ann::Chain, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, trainOnly2LastLayers::Bool;
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.001, minLossChange::Real=1e-7, lossChangeWindowSize::Int=5)
    #
    # Codigo a desarrollar
    #
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



