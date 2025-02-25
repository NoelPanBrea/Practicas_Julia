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
    """
    Los datos tienen que estar en Float32
    Si se pasa un validationDataset, la RNA devuelta es la de mejor
    error de validacion
    Si no se pasa un validationDataset, funciona igual que el Ej2 
    Hay que devolver una tupla de 4 elementos, 1º RNA entrenada, 2º train_losses, 3º val_losses, 4º test_losses
    """
    ann = buildClassANN(size(dataset[1], 2), topology, size(dataset[2], 2), transferFunctions = transferFunctions);

    dataset = (permutedims(convert(AbstractArray{Float32, 2}, dataset[1])), permutedims(dataset[2]));

    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y);
    
    train_losses = [loss(ann, dataset[1], dataset[2])];
    val_losses = Float32[];
    test_losses = Float32[];
    
    opt_state = Flux.setup(Adam(learningRate), ann);

    # Parada temprana
    best_val_loss = Inf32
    best_epoch = 0
    best_model = deepcopy(ann)

    if validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2))
        valid_data = (permutedims(convert(AbstractArray{Float32,2},valid_data[1])), permutedims(valid_data[2]));
        push!(val_losses, loss(ann, valid_data[1], valid_data[2]));
        best_val_loss = val_losses[1];
    end;
    if testDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2))
        test_data = (permutedims(convert(AbstractArray{Float32,2},test_data[1])), permutedims(test_data[2]));
        push!(test_losses, loss(ann, test_data[1], test_data[2]));
    end;

    cnt = 0
    while cnt < maxEpochs && train_losses[end] > minLoss && (cnt - best_epoch < maxEpochsVal)
        cnt += 1;
        Flux.train!(loss, ann, [dataset], opt_state);
        push!(train_losses, loss(ann, dataset[1], dataset[2]));

        if validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2))
            val_loss = loss(ann, valid_data[1], valid_data[2]);
            push!(validationLosses, val_loss);

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

    if validationDataset[1] != Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2))
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
    #
    # Codigo a desarrollar
    #
end;
