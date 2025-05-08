using DelimitedFiles
using Statistics
using Random
using Plots

# Importar todas las funciones del primer fichero
include("soluciones.jl")

# Función para probar diferentes valores de profundidad máxima en árboles de decisión
function testDifferentDepths(inputs, targets, depthValues)
    results = Dict()
    bestAccuracy = 0.0
    bestDepth = 0
    
    # Crear índices para validación cruzada
    crossValidationIndices = crossvalidation(oneHotEncoding(targets), 5)  # Usando 5 folds
    
    for depth in depthValues
        println("Probando con profundidad máxima = ", depth)
        
        # Configurar hiperparámetros para árboles de decisión
        hyperparameters = Dict("max_depth" => depth)
        
        # Realizar validación cruzada
        ((testAccuracy_mean, testAccuracy_std), 
         (testErrorRate_mean, testErrorRate_std), 
         (testRecall_mean, testRecall_std), 
         (testSpecificity_mean, testSpecificity_std), 
         (testPrecision_mean, testPrecision_std), 
         (testNPV_mean, testNPV_std), 
         (testF1_mean, testF1_std), 
         testConfusionMatrix) = modelCrossValidation(:DecisionTreeClassifier, hyperparameters, (inputs, targets), crossValidationIndices)
        
        # Guardar resultados
        results[depth] = (
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
            bestDepth = depth
        end
        
        # Imprimir resultados para este valor de profundidad
        println("  Accuracy: $(testAccuracy_mean) ± $(testAccuracy_std)")
        println("  F1 Score: $(testF1_mean) ± $(testF1_std)")
        println("  ---------------------------------------------")
    end
    
    println("\nMejor profundidad máxima: ", bestDepth)
    println("Con precisión: ", bestAccuracy)
    
    return results, bestDepth, bestAccuracy
end

# Cargar los datos
println("Cargando el dataset...")
dataset = readdlm("./Entrega/optical+recognition+of+handwritten+digits/optdigits.full", ',')

# Extraer entradas y salidas
inputs = Float32.(dataset[:,1:64])
targets = dataset[:,65]

# Opcionalmente, normalizar los datos (a diferencia de kNN, los árboles no son sensibles a la escala)
# inputs_normalized = normalizeMinMax(inputs)
# println("Datos normalizados")

# Definir valores de profundidad a probar
depthValues = [2, 4, 6, 8, 10, 12]

println("Iniciando pruebas con diferentes valores de profundidad para árboles de decisión...")
results, bestDepth, bestAccuracy = testDifferentDepths(inputs, targets, depthValues)

# Visualizar resultados
accuracies = [results[d].accuracy[1] for d in depthValues]
f1_scores = [results[d].f1[1] for d in depthValues]

# Crear gráfico de resultados
plt = plot(depthValues, accuracies, 
    label="Accuracy", 
    xlabel="Profundidad máxima", 
    ylabel="Valor métrica",
    title="Rendimiento árbol de decisión según profundidad",
    marker=:circle)
plot!(plt, depthValues, f1_scores, label="F1 Score", marker=:square)

# Guardar gráfico
savefig(plt, "resultados_arbol_profundidad.png")
println("Gráfico de resultados guardado como 'resultados_arbol_profundidad.png'")

# Mostrar detalles del mejor modelo
println("\nDetalles del mejor modelo (profundidad = $bestDepth):")
println("Accuracy: $(results[bestDepth].accuracy[1]) ± $(results[bestDepth].accuracy[2])")
println("Error Rate: $(results[bestDepth].error_rate[1]) ± $(results[bestDepth].error_rate[2])")
println("Recall: $(results[bestDepth].recall[1]) ± $(results[bestDepth].recall[2])")
println("Specificity: $(results[bestDepth].specificity[1]) ± $(results[bestDepth].specificity[2])")
println("Precision: $(results[bestDepth].precision[1]) ± $(results[bestDepth].precision[2])")
println("NPV: $(results[bestDepth].npv[1]) ± $(results[bestDepth].npv[2])")
println("F1 Score: $(results[bestDepth].f1[1]) ± $(results[bestDepth].f1[2])")
println("Matriz de confusión:")
display(results[bestDepth].confusion_matrix)

# Evaluar el impacto de la poda (opcional)
println("\nEvaluando el impacto de la limitación de profundidad (poda)...")
pruningImpact = (results[maximum(depthValues)].accuracy[1] - results[4].accuracy[1]) / results[maximum(depthValues)].accuracy[1] * 100
println("Diferencia de precisión entre profundidad máxima y profundidad media: $(round(pruningImpact, digits=2))%")

# Conclusión
if pruningImpact > 5
    println("Conclusión: La limitación de profundidad tiene un impacto significativo en el rendimiento (>5%).")
elseif pruningImpact > 0
    println("Conclusión: La limitación de profundidad mejora levemente el rendimiento.")
else
    println("Conclusión: La limitación de profundidad no aporta mejoras significativas para este dataset.")
end