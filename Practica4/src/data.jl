
function CreateDataset(datasetFolder::String)
    files = fileNamesFolder(datasetFolder, "csv");
    datasets_list = loadDataset.(files, dataset_folder);
    merged_data = DataFrame();
    for dataset in datasets_list
        merged_data = vcat(merged_data, dataset);
    end;
    return merged_data;
end;

function fileNamesFolder(folderName::String, extension::String)
    isdir(folderName) || return String[]
    extU = uppercase(extension)
    files = sort(filter(f -> endswith(uppercase(f), ".$extU"), readdir(folderName)))
    return map(f -> first(splitext(f)), files)
end


function loadDataset(datasetName::String, datasetFolder::String)
    fname = datasetName * ".csv"
    fpath = abspath(joinpath(datasetFolder, fname))
    isfile(fpath) || return nothing

    data = CSV.File(fpath, header=true) |> DataFrame

    return data
end;

function OneHotEncoding(labels::AbstractArray{<:Any,1})
    classes = convert(AbstractArray{<:Any, 1}, unique(labels));
    if length(classes) > 2
        return  convert(BitArray{2}, hcat([label .== classes for label in labels]...)');
    else
        return reshape(convert(AbstractArray{Bool,1}, labels .== classes.first()), :, 1);
    end;
end;

function holdOut(N::Int, P::Real)
    permutation = Random.randperm(N);
    test_index = permutation[1:(convert(Int64, (round(N*P))))];
    train_index = permutation[(convert(Int64, (round(N*P))) + 1):end];
    index = (train_index, test_index);
    return index;
end

function crossvalidation(n::Int64, k::Int64)
    v = 1:k;
    v_repeated = repeat(v, convert(Int64, ceil(n/k)));
    v_sliced = v_repeated[1:n];
    return Random.shuffle!(v_sliced);
end;

function crossvalidation(individuals::AbstractArray{<:Any,1}, k::Int64)
    n_ins = size(individuals, 1);
    n_inv = unique(individuals);
    indexes = zeros(Int64, n_ins);
    inv_indx = crossvalidation(length(n_inv), k)
    for i in 1:k
        inv_pos = findall(x -> x == i, inv_indx);
        indexes[findall(x -> any(x .== n_inv[inv_pos]), individuals)] .= i
    end;
    return indexes;
end;