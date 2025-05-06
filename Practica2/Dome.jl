using DelimitedFiles
using Statistics
using Random
using Plots

# Importar todas las funciones del primer fichero
include("soluciones.jl")

# Función para probar diferentes valores de nodos en DoME
function testDifferentNodes(inputs, targets, nodeValues)
    results = Dict()
    bestAccuracy = 0.0
    bestNodes = 0
    
    # Crear índices para validación cruzada
    crossValidationIndices = crossvalidation(oneHotEncoding(targets), 10)
    
    for nodes in nodeValues
        println("Probando con maximumNodes = ", nodes)
        
        # Configurar hiperparámetros para DoME
        hyperparameters = Dict("maximumNodes" => nodes)
        
        # Realizar validación cruzada
        ((testAccuracy_mean, testAccuracy_std), 
         (testErrorRate_mean, testErrorRate_std), 
         (testRecall_mean, testRecall_std), 
         (testSpecificity_mean, testSpecificity_std), 
         (testPrecision_mean, testPrecision_std), 
         (testNPV_mean, testNPV_std), 
         (testF1_mean, testF1_std), 
         testConfusionMatrix,
         _) = modelCrossValidation(:DoME, hyperparameters, (inputs, targets), crossValidationIndices)
        
        # Guardar resultados
        results[nodes] = (
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
            bestNodes = nodes
        end
        
        # Imprimir resultados para este valor de nodos
        println("  Accuracy: $(testAccuracy_mean) ± $(testAccuracy_std)")
        println("  F1 Score: $(testF1_mean) ± $(testF1_std)")
        println("  ---------------------------------------------")
    end
    
    println("\nMejor número de nodos: ", bestNodes)
    println("Con precisión: ", bestAccuracy)
    
    return results, bestNodes, bestAccuracy
end

# Cargar los datos
println("Cargando el dataset...")
dataset = readdlm("Practica2/optical+recognition+of+handwritten+digits/optdigits.full", ',')

# Extraer entradas y salidas
inputs = Float32.(dataset[:,1:64])
targets = dataset[:,65]

# Probar normalización de datos (opcional)
# Comentar/descomentar según se necesite
# inputs_normalized = normalizeMinMax(inputs)
# println("Datos normalizados")

# Definir valores de nodos a probar
# Puedes ajustar este rango según los resultados iniciales
nodeValues = [5, 10, 15, 20, 25, 30, 35, 40]

println("Iniciando pruebas con diferentes valores de nodos para DoME...")
results, bestNodes, bestAccuracy = testDifferentNodes(inputs, targets, nodeValues)

# Visualizar resultados
accuracies = [results[n].accuracy[1] for n in nodeValues]
f1_scores = [results[n].f1[1] for n in nodeValues]

# Crear gráfico de resultados
plt = plot(nodeValues, accuracies, 
    label="Accuracy", 
    xlabel="Número de nodos", 
    ylabel="Valor métrica",
    title="Rendimiento DoME según número de nodos",
    marker=:circle)
plot!(plt, nodeValues, f1_scores, label="F1 Score", marker=:square)

# Guardar gráfico
savefig(plt, "resultados_dome_nodos.png")
println("Gráfico de resultados guardado como 'resultados_dome_nodos.png'")

# Mostrar detalles del mejor modelo
println("\nDetalles del mejor modelo (maximumNodes = $bestNodes):")
println("Accuracy: $(results[bestNodes].accuracy[1]) ± $(results[bestNodes].accuracy[2])")
println("Error Rate: $(results[bestNodes].error_rate[1]) ± $(results[bestNodes].error_rate[2])")
println("Recall: $(results[bestNodes].recall[1]) ± $(results[bestNodes].recall[2])")
println("Specificity: $(results[bestNodes].specificity[1]) ± $(results[bestNodes].specificity[2])")
println("Precision: $(results[bestNodes].precision[1]) ± $(results[bestNodes].precision[2])")
println("NPV: $(results[bestNodes].npv[1]) ± $(results[bestNodes].npv[2])")
println("F1 Score: $(results[bestNodes].f1[1]) ± $(results[bestNodes].f1[2])")
println("Matriz de confusión:")
display(results[bestNodes].confusion_matrix)