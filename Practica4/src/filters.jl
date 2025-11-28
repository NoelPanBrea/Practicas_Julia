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


function anova()
end;