# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    """
    Esta funcion acepta dos vectores de igual longitud, el primero con las salidas 
    obtenidas por un modelo outputs y el segundo con las salidas deseadas targets, 
    ambos de tipo AbstractArray{Bool,1} y devuelva una tupla con los siguientes 
    valores: (Valor de precisión, Tasa de fallo, Sensibilidad, Especificidad, Valor 
    predictivo positivo, Valor predictivo negativo, F1-score, Matriz de confusión(Array{Int64,2}))
    """
    
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    """
    Esta función de nombre igual que la anterior, cuyo primer parámetro, en lugar de ser un
    vector de valores booleanos, es un vector de valores reales (de tipo AbstractArray{<:Real}), y
    con un tercer parámetro opcional que tenga un umbral, con un valor por defecto, y los utiliza
    para aplicar la función anterior y devolver los mismos valores
    """


end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1},
    targets::AbstractArray{Bool,1})
    ann = confusionMatrix(outputs,targets);
    print("Valor de precisión: ",ann[1]);
    print("Tasa de fallo: ", ann[2]);
    print("Sensibilidad: ", ann[3]);
    print("Especificidad: ", ann[4]);
    print("Valor predictivo positivo: ", ann[5]);
    print("F1-score: ", ann[6]);
    print("Matriz de confusión: ", ann[7]);

end;
    
function printConfusionMatrix(outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1}; threshold::Real=0.5) 
    ann = confusionMatrix(outputs, targets, threshold=threshold);
    print("Valor de precisión: ",ann[1]);
    print("Tasa de fallo: ", ann[2]);
    print("Sensibilidad: ", ann[3]);
    print("Especificidad: ", ann[4]);
    print("Valor predictivo positivo: ", ann[5]);
    print("F1-score: ", ann[6]);
    print("Matriz de confusión: ", ann[7]);

end;