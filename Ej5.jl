using Random
using Random:seed!
#=
Corregir crossvalidation, da índices erróneos 
Posible fallo en línea 8 de crossvalidation(N::Int64, k::Int64)
=#
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
