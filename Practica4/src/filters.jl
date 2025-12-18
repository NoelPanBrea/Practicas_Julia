include("firmas.jl")

using Statistics
using StatsBase
using MLJ
using MLJLinearModels

function kendall(dataset::Tuple{DataFrame, BitArray})
    targets = sum.(findall.(eachrow(dataset[2])));
    inputs = Matrix(dataset[1]);
    input_size = size(inputs, 1);
    CP = zeros(size(inputs, 2));
    for (n, (i, x)) in enumerate(zip(1:input_size, targets))
        i = inputs[i, :]
        for (j, y) in zip(n:input_size, targets[n+1:end])
            j = inputs[j, :]
            t1 = (i .< j) .& (x < y)
            t2 = (i .> j) .& (x > y)
            t3 = convert(Array{Int64}, t1 .|| t2)
            t3[findall(x -> x == 0, t3)] .-= 1
            CP += t3
        end;
    end;
    return CP ./ (0.5 * size(inputs, 1) * size(inputs, 1))
end;



# function anova(dataset::Tuple{DataFrame, BitArray})
#     inputs, labels = dataset; 
#     means = mean.(eachcol(inputs));
#     targets = [sum(findall(row)) for row in eachrow(labels)]
#     classes = unique(targets);
#     mean_class = Dict{Int, Vector{Float64}}();
#     for class in classes
#         index = findall(x -> x == class, targets);
#         mean_class[class] = mean.(eachcol(inputs[index, :])); 
#     end;

#     sst = sum(sum([(col .- mn) .^ 2 for (mn, col) in zip(means, eachcol(inputs))]));
#     sse = 0;
#     for class in classes
#         index = findall(x -> x == class, targets)
#         sse += sum(sum([(col .- mc).^2 for (col, mc) in zip(eachcol(inputs[index, :]), mean_class[class])]))
#     end;

#     return (sst - sse) / (length(classes) - 1) / (sse / (size(inputs, 1) - length(classes)));
# end;


function anova(dataset::Tuple{DataFrame, BitArray})
    inputs, labels = dataset; 
    targets = [sum(findall(row)) for row in eachrow(labels)]
    classes = unique(targets)

    scores = Float64[]

    for col in eachcol(inputs)
        col_mean = mean(col)
        sst = 0.0
        sse = 0.0

        for c in classes
            idx = findall(targets .== c)
            class_mean = mean(col[idx])
            sst += length(idx) * (class_mean - col_mean)^2
            sse += sum((col[idx] .- class_mean).^2)
        end

        push!(scores, sst / sse)
    end

    return scores
end



function pearson(dataset::Tuple{DataFrame, BitArray}; threshold=0.1)
    inputs, labels = dataset
    targets = [sum(findall(row)) for row in eachrow(labels)]
    correlations = Dict{String, Float64}()
    
    for (col_name, col) in zip(names(inputs), eachcol(inputs))
        # he creado las variables numerador y denominador para que no quede muy larga la expresión de correlación
        numerator = (sum((col .- mean(col)) .* (targets .- mean(targets))))
        denominator = (sqrt(sum((col .- mean(col)).^2)) * sqrt(sum((targets .- mean(targets)).^2)))
        correlations[col_name] =  numerator / denominator
    end

    filtered_cols = [col for col in keys(correlations) if abs(correlations[col]) ≥ threshold]

    return inputs[:, filtered_cols], correlations
end



# función auxiliar de spearman para calcular ranks
function compute_ranks(x::AbstractVector)
    sorted_idx = sortperm(x)               
    ranks = similar(x, Float64)
    for (rank, idx) in enumerate(sorted_idx)
        ranks[idx] = rank
    end
    return ranks
end

function spearman(dataset::Tuple{DataFrame, BitArray})
    inputs, labels = dataset
    targets = [sum(findall(row)) for row in eachrow(labels)]
    scores = Dict{String, Float64}()
    for (col_name, col) in zip(names(inputs), eachcol(inputs))
        scores[col_name] = cor(compute_ranks(col), compute_ranks(targets))
    end
    return scores
end




function rfe_logistic(dataset::Tuple{DataFrame, BitArray})
    inputs, labels = dataset
    targets = [sum(findall(row)) for row in eachrow(labels)]

    features = collect(names(inputs))
    targets_categorical = categorical(string.(targets))

    while length(features) > 1
        model = LogisticClassifier()
        mach = machine(model, coalesce.(inputs[:, features], 0.0), targets_categorical)
        MLJ.fit!(mach)

        coefs = abs.(mach.fitresult.coefs[1][2:end])
        k = ceil(Int, length(features) / 2)

        idx = sortperm(coefs, rev=true)[1:k]
        features = features[idx]
    end

    return features
end



function mutual_information(dataset::Tuple{DataFrame, BitArray})
end