include("firmas.jl")

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



function anova(dataset::Tuple{DataFrame, BitArray})
    inputs, labels = dataset; 
    means = mean.(eachcol(inputs));
    targets = sum.(findall.(eachrow(labels)));
    classes = unique(targets);
    mean_class = [];
    print(classes)
    # print(targets)
    for class in classes
        index = findall(x -> x == class, labels);
        println(index);
        push!(mean_class, mean.(eachcol(inputs[index, :]))); 
    end;
    println(mean_class)

    sst = sum(sum([(col .- mn) .^ 2 for (mn, col) in zip(means, eachcol(inputs))]));
    print(sst)
    sse = 0;
    for c in classes
        index = findall(row -> row == c, eachrow(labels));
        println(size([(col .- mean_class[c]) .^ 2 for col in eachcol(inputs[index, :])]))
        sse += sum(sum([(col .- mean_class[c]) .^ 2 for col in eachcol(inputs[index, :])]));
    end;
    return (sst - sse) / (length(classes) - 1) / (sse / (size(inputs, 1) - length(classes)));
end;