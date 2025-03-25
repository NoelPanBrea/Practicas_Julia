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

    if k >= 1
        error("ERROR: k debe ser al menos 1");
    end;
    
    v = zeros(Int64, len);
    
    num_true = sum(targets);
    num_false = len - num_true;

    if num_true < k || num_false < k
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

    if k >= 1
        error("ERROR: k debe ser al menos 1");
    end;

    v = zeros(Int64, num_patterns);

    for j in 1:num_classes
        class_positions = findall(targets[:,j]);

        num_class_patterns = length(class_positions)
        if num_class_patterns < k
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
    #
    # Codigo a desarrollar
    #
end;
