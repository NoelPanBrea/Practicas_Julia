using Plots
using StatsBase
using Statistics
using Distributions

"""
    create_cd_diagram(methods, performances; α=0.05)

Crea un diagrama de diferencia crítica (CD) para comparar varios métodos.

Parámetros:
- `methods`: Vector de nombres de los métodos
- `performances`: Matriz donde cada fila son los resultados de un dataset y cada columna un método
- `α`: Nivel de significancia para el test de Nemenyi (por defecto 0.05)

Nota: Las performances pueden ser cualquier métrica (error, exactitud, etc.). 
Si valores más bajos son mejores (ej. error), establece `lower_is_better=true`.
Si valores más altos son mejores (ej. exactitud), establece `lower_is_better=false`.
"""
function create_cd_diagram(methods, performances; α=0.05, lower_is_better=true, title="", figsize=(800, 400))
    n_methods = length(methods)
    n_datasets = size(performances, 1)
    
    # Calcular rangos para cada dataset
    ranks = zeros(n_datasets, n_methods)
    for i in 1:n_datasets
        if lower_is_better
            # Para métricas donde valores más bajos son mejores (como error)
            ranks[i, :] = tiedrank(performances[i, :])
        else
            # Para métricas donde valores más altos son mejores (como exactitud)
            ranks[i, :] = tiedrank(1.0 .- performances[i, :])
        end
    end
    
    # Calcular rangos medios
    avg_ranks = vec(mean(ranks, dims=1))
    
    # Calcular diferencia crítica según el test de Nemenyi
    q_alpha = quantile(StudentizedRange(n_methods), 1 - α)
    cd = q_alpha * sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    
    # Ordenar métodos por rango medio
    sorted_indices = sortperm(avg_ranks)
    sorted_methods = methods[sorted_indices]
    sorted_ranks = avg_ranks[sorted_indices]
    
    # Crear el gráfico
    plt = plot(
        xlim=(0.5, n_methods + 0.5),
        ylim=(0, n_methods + 1),
        xticks=1:n_methods,
        yticks=[],
        grid=false,
        legend=false,
        size=figsize,
        margin=10Plots.mm,
        title=title
    )
    
    # Dibujar línea horizontal superior para la escala
    plot!(plt, [1, n_methods], [n_methods + 0.5, n_methods + 0.5], color=:black, linewidth=1)
    
    # Dibujar líneas verticales para la escala
    for i in 1:n_methods
        plot!(plt, [i, i], [n_methods + 0.3, n_methods + 0.7], color=:black, linewidth=1)
    end
    
    # Dibujar CD en la parte superior
    cd_start = 1.5
    cd_end = cd_start + cd
    plot!(plt, [cd_start, cd_end], [n_methods + 1, n_methods + 1], color=:black, linewidth=2)
    annotate!(plt, [(cd_start + cd/2, n_methods + 1.2, "CD = $(round(cd, digits=2))")])
    
    # Dibujar líneas horizontales para cada algoritmo y añadir nombres
    for (i, (method, rank)) in enumerate(zip(sorted_methods, sorted_ranks))
        y_pos = n_methods - i + 1
        # Línea desde el eje y hasta el rango
        plot!(plt, [1, rank], [y_pos, y_pos], color=:black, linewidth=1)
        # Añadir nombre del método
        annotate!(plt, [(0.8, y_pos, method, :right, :middle, 9)])
    end
    
    # Identificar y dibujar grupos de métodos no significativamente diferentes
    groups = []
    current_group = [1]
    
    for i in 2:length(sorted_ranks)
        if sorted_ranks[i] - sorted_ranks[current_group[1]] <= cd
            push!(current_group, i)
        else
            push!(groups, copy(current_group))
            current_group = [i]
        end
    end
    push!(groups, current_group)
    
    # Dibujar líneas horizontales para grupos
    for (group_idx, group) in enumerate(groups)
        if length(group) > 1
            min_rank = sorted_ranks[group[1]]
            max_rank = sorted_ranks[group[end]]
            y_pos = n_methods - group[1] + 1 + 0.2
            plot!(plt, [min_rank, max_rank], [y_pos, y_pos], color=:black, linewidth=2)
        end
    end
    
    return plt
end

# Ejemplo de uso con 5 métodos
methods = ["Método A", "Método B", "Método C", "Método D", "Método E"]

# Matriz de rendimiento simulada para 10 datasets (filas) y 5 métodos (columnas)
# Podría ser error, exactitud, F1-score, etc.
# Aquí simulamos alguna métrica de error (menor es mejor)
performances = [
    0.15 0.21 0.19 0.25 0.22;
    0.12 0.18 0.15 0.20 0.17;
    0.19 0.15 0.22 0.18 0.20;
    0.14 0.21 0.16 0.23 0.19;
    0.16 0.19 0.18 0.22 0.20;
    0.13 0.18 0.17 0.21 0.19;
    0.17 0.20 0.15 0.22 0.18;
    0.15 0.17 0.19 0.23 0.21;
    0.18 0.16 0.20 0.24 0.19;
    0.16 0.19 0.17 0.21 0.18
]

# Crear el diagrama CD
cd_diagram = create_cd_diagram(
    methods, 
    performances, 
    α=0.05, 
    lower_is_better=true,  # Si tu métrica es error (menor es mejor)
    title="Comparación de Métodos ML"
)

# Guardar la figura
savefig(cd_diagram, "cd_diagram.png")

# Para mostrarla en un entorno interactivo
display(cd_diagram)
