function kendall(dataset::Tuple{DataFrame, BitArray})
    CP = 0;
    targets = findall.(eachrow(dataset[2]));
    inputs = dataset[1];
    for (i, x) in zip(eachrow(inputs), targets)
        for (j, y) in zip(eachrow(inputs), targets)
            t1 = any(i .< j & x < y)
            t2 = any(i .> j & x > y)
            if j!=i & (t1 || t2)
                CP += count(t1) + count(t2)
            end;
        end;
    end;
    return CP
end;



function anova(dataset::Tuple{DataFrame, BitArray})
    inputs, labels = dataset; 
    means = mean.(inputs[:, eachcol(inputs)]); 
    targets = findall(eachrow[labels]);
    classes = unique(targets);
    mean_class = [];
    for class in classes
        index = findall(row -> row == class, eachrow(labels)); 
        push!(mean_class, mean(inputs[index, eachcol(inputs)])); 
    end;

    sst = sum((inputs[:, eachcol(inputs)] .- means) .^ 2);
    sse = 0;
    for c in classes
        index = findall(row -> row == class, eachrow(labels)); 
        sse += sum([sum((inputs[index, :] .- mean_class[c]).^2)]);
    end;
    f = (sst - sse) / (length(classes) - 1) / (sse / (size(inputs, 1) - length(classes)));
end;