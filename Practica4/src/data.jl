include("firmas.jl")

function CreateDataset(datasetFolder::String)
    files = fileNamesFolder(datasetFolder, "csv")
    _, merged_data = vcat(loadDataset.(files, dataset_folder))
    return merged_data
end

dataset_folder = "Practica4/dataset"
dataset = CreateDataset(dataset_folder)
writedlm("Practica4/createddataset/merged.csv", dataset, ',')
