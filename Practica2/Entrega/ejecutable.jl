# Script para ejecutar todos los experimentos en orden
println("=========================================================");
println("REPRODUCCIÓN DE EXPERIMENTOS CON ÍNDICES PREDEFINIDOS");
println("=========================================================");

# 1. Primero verificamos si ya existen los índices de CV
if !isfile("cv_indices.txt");
    println("\n1. Generando índices de validación cruzada...");
    include("index.jl");
else
    println("\n1. Usando índices de validación cruzada existentes");
end;

# 2. Ejecutar el script básico de validación cruzada
println("\n2. Ejecutando validación cruzada básica...");
include("crossvalidation.jl");

# 3. Ejecutar experimento completo
println("\n3. Ejecutando experimento completo con visualizaciones detalladas...");
include("visualizacion.jl");

println("\n=========================================================");
println("TODOS LOS EXPERIMENTOS COMPLETADOS");
println("Resultados guardados como archivos PNG");
println("=========================================================");

"""
------TODO----------------
- Añadir la gráfica que dijo el profe
- Mejorar la optimización del código (muy repetitivo)
- Adaptar el código a lo que devuelve la p1
- Mejor forma para añadir los valores encontrados en la investigación (no solo uno)
- Mejor forma para las direcciones de los archivos (dataset e indices)
- Hacer lo mismo que con ANN con todos los modelos (una vez arreglado el trozo de código repetido para el entrenamiento)

-----TODO OPCIONALES----------
- Mejor forma de hacer las tablas (más bonitas)
- Añadir más gráficas y tablas (las que hablamos en clase)
- Intentar hacerlo más legible y optimizado (usar eachindex, no usar ?,...)
"""