# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    permutation = Random.randperm(N);
    test_index = permutation[1:(convert(Int64, (round(N*P))))];
    train_index = permutation[(convert(Int64, (round(N*P))) + 1):end];
    index = (train_index, test_index);
    return index;
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    trainval_test = holdOut(N, Ptest);
    train_index = trainval_test[1]; 
    test_index = trainval_test[2];
    
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
    #=
    Los datos tienen que estar en Float32
    Si se pasa un validationDataset, la RNA devuelta es la de mejor
    error de validacion
    Si no se pasa un validationDataset, funciona igual que el Ej2 
    Hay que devolver una tupla de 4 elementos, 1º RNA entrenada, 2º train_losses, 3º val_losses, 4º test_losses
    =#
    ann = buildClassANN(size(dataset[1], 2), topology, size(dataset[2], 2), transferFunctions = transferFunctions);

    dataset = (permutedims(convert(AbstractArray{Float32, 2}, dataset[1])), permutedims(dataset[2]));

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
