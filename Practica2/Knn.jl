using DelimitedFiles
using Statistics
using Random
using Plots

# Importar todas las funciones del primer fichero
include("soluciones.jl")

# Función para probar diferentes valores de k en kNN
function testDifferentKValues(inputs, targets, kValues)
    results = Dict()
    bestAccuracy = 0.0
    bestK = 0
    
    # Crear índices para validación cruzada
    crossValidationIndices = crossvalidation(oneHotEncoding(targets), 5)  # Usando 5 folds como en tu index.jl
    
    for k in kValues
        println("Probando con k = ", k)
        
        # Configurar hiperparámetros para kNN
        hyperparameters = Dict("n_neighbors" => k)
        
        # Realizar validación cruzada
        ((testAccuracy_mean, testAccuracy_std), 
         (testErrorRate_mean, testErrorRate_std), 
         (testRecall_mean, testRecall_std), 
         (testSpecificity_mean, testSpecificity_std), 
         (testPrecision_mean, testPrecision_std), 
         (testNPV_mean, testNPV_std), 
         (testF1_mean, testF1_std), 
         testConfusionMatrix) = modelCrossValidation(:KNeighborsClassifier, hyperparameters, (inputs, targets), crossValidationIndices)
        
        # Guardar resultados
        results[k] = (
            accuracy = (testAccuracy_mean, testAccuracy_std),
            error_rate = (testErrorRate_mean, testErrorRate_std),
            recall = (testRecall_mean, testRecall_std),
            specificity = (testSpecificity_mean, testSpecificity_std),
            precision = (testPrecision_mean, testPrecision_std),
            npv = (testNPV_mean, testNPV_std),
            f1 = (testF1_mean, testF1_std),
            confusion_matrix = testConfusionMatrix
        )
        
        # Actualizar mejor valor si es necesario
        if testAccuracy_mean > bestAccuracy
            bestAccuracy = testAccuracy_mean
            bestK = k
        end
        
        # Imprimir resultados para este valor de k
        println("  Accuracy: $(testAccuracy_mean) ± $(testAccuracy_std)")
        println("  F1 Score: $(testF1_mean) ± $(testF1_std)")
        println("  ---------------------------------------------")
    end
    
    println("\nMejor valor de k: ", bestK)
    println("Con precisión: ", bestAccuracy)
    
    return results, bestK, bestAccuracy
end

# Cargar los datos
println("Cargando el dataset...")
dataset = readdlm("./Entrega/optical+recognition+of+handwritten+digits/optdigits.full", ',')

# Extraer entradas y salidas
inputs = Float32.(dataset[:,1:64])
targets = dataset[:,65]

# Normalizar los datos (recomendado para kNN)
inputs_normalized = normalizeMinMax(inputs)
println("Datos normalizados")

# Definir valores de k a probar
# Suele ser mejor usar valores impares para evitar empates
kValues = [1, 3, 5, 7, 9, 11]

println("Iniciando pruebas con diferentes valores de k para kNN...")
results, bestK, bestAccuracy = testDifferentKValues(inputs_normalized, targets, kValues)

# Visualizar resultados
accuracies = [results[k].accuracy[1] for k in kValues]
f1_scores = [results[k].f1[1] for k in kValues]

# Crear gráfico de resultados
plt = plot(kValues, accuracies, 
    label="Accuracy", 
    xlabel="Valor de k", 
    ylabel="Valor métrica",
    title="Rendimiento kNN según valor de k",
    marker=:circle)
plot!(plt, kValues, f1_scores, label="F1 Score", marker=:square)

# Guardar gráfico
savefig(plt, "resultados_knn_k.png")
println("Gráfico de resultados guardado como 'resultados_knn_k.png'")

# Mostrar detalles del mejor modelo
println("\nDetalles del mejor modelo (k = $bestK):")
println("Accuracy: $(results[bestK].accuracy[1]) ± $(results[bestK].accuracy[2])")
println("Error Rate: $(results[bestK].error_rate[1]) ± $(results[bestK].error_rate[2])")
println("Recall: $(results[bestK].recall[1]) ± $(results[bestK].recall[2])")
println("Specificity: $(results[bestK].specificity[1]) ± $(results[bestK].specificity[2])")
println("Precision: $(results[bestK].precision[1]) ± $(results[bestK].precision[2])")
println("NPV: $(results[bestK].npv[1]) ± $(results[bestK].npv[2])")
println("F1 Score: $(results[bestK].f1[1]) ± $(results[bestK].f1[2])")
println("Matriz de confusión:")
display(results[bestK].confusion_matrix)