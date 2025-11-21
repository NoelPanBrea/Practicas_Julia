include("firmas.jl")

using CSV
using DataFrames


function CreateDataset(datasetFolder::String)
    files = fileNamesFolder(datasetFolder, "csv")
    datasets_list = loadDataset.(files, dataset_folder)
    merged_data = DataFrame()
    for dataset in datasets_list
    merged_data = vcat(merged_data, dataset)
    end

    return merged_data
end

dataset_folder = "Practica4/dataset"
dataset = CreateDataset(dataset_folder)
CSV.write("Practica4/createddataset/merged.csv", dataset)
