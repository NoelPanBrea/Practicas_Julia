using Random
using DelimitedFiles

include("73166321D_54157616E_48118254T_54152126Y.jl")  

Random.seed!(12345)

function load_example_dataset(filename::String)
    data = readdlm(filename, ',')
    
    inputs = data[:, 1:64]
    inputs = Float32.(inputs)
    targets = data[:,65]
    
    return (inputs, targets)
end

function main()

    dataset_file = "Entrega/optdigits.full"
    
    try
        println("Loading dataset from: ", dataset_file)
        dataset = load_example_dataset(dataset_file)
        println("Dataset loaded: $(size(dataset[1], 1)) patterns with $(size(dataset[1], 2)) features")
        
        k = 5
        println("Generating indices for $k-fold cross validation...")
        
        cv_indices = crossvalidation(dataset[2], k)
        
        test_indices = findall(x -> x == 1, cv_indices)
        
        indices_file = "Entrega/cv_indices.txt"
        writedlm(indices_file, test_indices)
        println("Test indices saved to: ", indices_file)
        
    catch e
        println("Error during index generation: ", e)
        return 1
    end
    
    return 0
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end