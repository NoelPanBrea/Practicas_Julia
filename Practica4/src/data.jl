include("firmas.jl")


function CreateDataset(datasetFolder::String)
    files = fileNamesFolder(datasetFolder, "csv")
    datasets_list = loadDataset.(files, dataset_folder)
    merged_data = DataFrame()
    for dataset in datasets_list
        merged_data = vcat(merged_data, dataset)
    end

    return merged_data
end