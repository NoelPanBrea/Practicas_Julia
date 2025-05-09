using Random
using DelimitedFiles

include("73166321D_54157616E_48118254T_54152126Y.jl");  # Para usar la función crossvalidation

# Fijar la semilla aleatoria para reproducibilidad
Random.seed!(12345);

# Cargar el dataset de ejemplo 
function load_example_dataset(filename::String)
    data = readdlm(filename, ',');
    
    inputs = data[:, 1:64];
    inputs = Float32.(inputs)
    targets = data[:,65];
    
    return (inputs, targets);
end;


function main()
    # Nombre del archivo del dataset
    dataset_file = "Entrega/optdigits.full";
    
    try
        println("Cargando dataset desde: ", dataset_file);
        dataset = load_example_dataset(dataset_file);
        println("Dataset cargado: $(size(dataset[1], 1)) patrones con $(size(dataset[1], 2)) características");
        
        # Número de folds para validación cruzada
        k = 5;
        println("Generando índices para $k-fold cross validation...");
        
        # Generar índices estratificados
        cv_indices = crossvalidation(dataset[2], k);
        
        # Guardar los índices en un archivo
        indices_file = "Entrega/cv_indices.txt";
        writedlm(indices_file, cv_indices);
        println("Índices guardados en: ", indices_file);
        
    catch e
        println("Error durante la generación de índices: ", e);
        return 1;
    end
    
    return 0;
end;