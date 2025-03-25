# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------
using Statistics
using Flux
using Flux.Losses
using Random
#=
Se calculan mal los acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix 
No sé si en binarias y en multiclase o si solo en unas 
SUPONGO binarias
=#

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    VP = sum(outputs .&& targets);
    VN = sum(.!outputs .&& .!targets);
    FP = sum(outputs .&& .!targets);
    FN = sum(.!outputs .&& targets);

    matriz_confusion = [VN FP; FN VP];

    if VP == 0 && FN == 0  
        sensibilidad = 1;
    else
        sensibilidad = VP / (FN + VP);
    end;
    
    if VP == 0 && FP == 0  
        valor_predictivo_positivo = 1;
    else
        valor_predictivo_positivo = VP / (VP + FP);
    end;

    if VN == 0 && FP == 0  
        especificidad = 1;
    else
        especificidad = VN / (FP + VN);
    end;

    if VN == 0 && FN == 0  
        valor_predictivo_negativo = 1  ;
    else
        valor_predictivo_negativo = VN / (VN + FN);
    end;

    precision = (VN + VP) / (VN + VP + FN + FP);
    tasa_error = (FN + FP) / (VN + VP + FN + FP);

    if valor_predictivo_positivo == 0 && sensibilidad == 0
        f1_score = 0;
    else
        f1_score = (2*valor_predictivo_positivo*sensibilidad)/ (valor_predictivo_positivo + sensibilidad);
    end;

    return (precision, tasa_error, sensibilidad, especificidad, valor_predictivo_positivo, valor_predictivo_negativo, f1_score, matriz_confusion)
    

end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    new_outputs = classifyOutputs(outputs, threshold = threshold);
    confusionMatrix(new_outputs, targets);
end;



function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    if size(outputs, 2) == 1 && size(targets, 2) == 1
        return confusionMatrix(outputs[:,1], targets[:,1]);
    end;

    if size(outputs, 2) > 2 && size(targets, 2) > 2
        num_classes = size(outputs, 2);
        sensibilidad = zeros(Float64, num_classes);
        especificidad = zeros(Float64, num_classes);
        valor_predictivo_positivo = zeros(Float64, num_classes);
        valor_predictivo_negativo = zeros(Float64, num_classes);
        f1_score = zeros(Float64, num_classes);

        
        for i in 1:num_classes
            outputs_class = outputs[:, i];
            targets_class = targets[:, i];
            
            stats = confusionMatrix(outputs_class, targets_class);
            _, _, sensibilidad[i], especificidad[i], valor_predictivo_positivo[i], 
            valor_predictivo_negativo[i], f1_score[i] = stats
        end;
  
        matriz_confusion = [sum((outputs[:, i] .== 1) .&& (targets[:, j] .== 1)) 
                            for i in 1:num_classes, j in 1:num_classes]

        instancias_clase = vec(sum(targets, dims=1));

        if weighted == true
            sensibilidad_media = sum(sensibilidad .* instancias_clase) / sum(instancias_clase);
            especificidad_media = sum(especificidad .* instancias_clase) / sum(instancias_clase);
            valor_predictivo_positivo_medio = sum(valor_predictivo_positivo .* instancias_clase) / sum(instancias_clase);
            valor_predictivo_negativo_medio = sum(valor_predictivo_negativo .* instancias_clase) / sum(instancias_clase);
            f1_score_medio = sum(f1_score .* instancias_clase) / sum(instancias_clase);
        else
            sensibilidad_media = mean(sensibilidad);
            especificidad_media = mean(especificidad);
            valor_predictivo_positivo_medio = mean(valor_predictivo_positivo);
            valor_predictivo_negativo_medio = mean(valor_predictivo_negativo);
            f1_score_medio = mean(f1_score);
        end;

        precision = accuracy(outputs, targets);
        tasa_error = 1 - precision;

        return (precision, tasa_error, sensibilidad_media, especificidad_media, valor_predictivo_positivo_medio, valor_predictivo_negativo_medio, f1_score_medio, matriz_confusion);
    end;
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    new_outputs = classifyOutputs(outputs, threshold = threshold);
    return confusionMatrix(new_outputs, targets, weighted = weighted);
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(label, classes) for label in vcat(targets, outputs)]));
    bool_outputs = oneHotEncoding(outputs);
    bool_targets = oneHotEncoding(targets);
    return confusionMatrix(bool_outputs, bool_targets, weighted = weighted);
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs));
    return confusionMatrix(outputs, targets, classes, weighted = weighted);
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = convert(AbstractArray{Float64,2}, trainingDataset[1]);
    trainingTargets = convert(AbstractArray{Bool,1}, trainingDataset[2]);
    testInputs = convert(AbstractArray{Float64,2}, testInputs);

    _, _, _, model = dome(trainingInputs, trainingTargets; maximumNodes = maximumNodes);

    return evaluateTree(model, testInputs);
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = convert(AbstractArray{Float64,2}, trainingDataset[1]);
    trainingTargets = convert(AbstractArray{Bool,2}, trainingDataset[2]);
    testInputs = convert(AbstractArray{Float64,2}, testInputs);

    if size(trainingTargets,2) == 1
        result = trainClassDoME((trainingInputs, vec(trainingTargets)), testInputs, maximumNodes);
        return reshape(result, :, 1);
    end;

    num_classes = size(trainingTargets, 2);
    result = zeros(Float64, size(testInputs,1), num_classes);

    for i in 1:num_classes
        class_results = trainClassDoME((trainingInputs, trainingTargets[:,i]), testInputs, maximumNodes);

        result[:,i] = class_results;
    end;

    return result
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    trainingInputs = convert(AbstractArray{Float64,2}, trainingDataset[1]);
    trainingTargets = trainingDataset[2];
    testInputs = convert(AbstractArray{Float64,2}, testInputs);

    classes = unique(trainingTargets);

    testOutputs = Array{eltype(trainingTargets),1}(undef, size(testInputs, 1));
    testOutputsDoME = trainClassDoME((trainingInputs, oneHotEncoding(trainingTargets, classes)), testInputs, maximumNodes);
    testOutputsBool = classifyOutputs(testOutputsDoME; threshold=0);

    num_classes = length(classes);

    if num_classes <= 2
        testOutputsBool = vec(testOutputsBool);
        testOutputs[testOutputsBool] .= classes[1];

        if num_classes == 2
            testOutputs[.!testOutputsBool] .= classes[2];
        end;
    else
        for i in 1:num_classes
            testOutputs[testOutputsBool[:,i]] .= classes[i];
        end;
    end;

    return testOutputs;
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


function printConfusionMatrix(outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
    ann = confusionMatrix(outputs, targets, weighted);
    print("Valor de precisión: ",ann[1]);
    print("Tasa de fallo: ", ann[2]);
    print("Sensibilidad: ", ann[3]);
    print("Especificidad: ", ann[4]);
    print("Valor predictivo positivo: ", ann[5]);
    print("F1-score: ", ann[6]);
    print("Matriz de confusión: ", ann[7]);
end;
    
    
function printConfusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
    ann = confusionMatrix(outputs, targets, weighted);
    print("Valor de precisión: ",ann[1]);
    print("Tasa de fallo: ", ann[2]);
    print("Sensibilidad: ", ann[3]);
    print("Especificidad: ", ann[4]);
    print("Valor predictivo positivo: ", ann[5]);
    print("F1-score: ", ann[6]);
    print("Matriz de confusión: ", ann[7]);

end;


function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1},
    classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(label, classes) for label in vcat(targets, outputs)]));
    bool_outputs = oneHotEncoding(outputs, classes);
    bool_targets = oneHotEncoding(targets, classes);
    ann = confusionMatrix(bool_outputs, bool_targets, weighted);
    print("Valor de precisión: ",ann[1]);
    print("Tasa de fallo: ", ann[2]);
    print("Sensibilidad: ", ann[3]);
    print("Especificidad: ", ann[4]);
    print("Valor predictivo positivo: ", ann[5]);
    print("F1-score: ", ann[6]);
    print("Matriz de confusión: ", ann[7]);

end;


function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs));
    ann = confusionMatrix(outputs, classes, weighted);
    print("Valor de precisión: ",ann[1]);
    print("Tasa de fallo: ", ann[2]);
    print("Sensibilidad: ", ann[3]);
    print("Especificidad: ", ann[4]);
    print("Valor predictivo positivo: ", ann[5]);
    print("F1-score: ", ann[6]);
    print("Matriz de confusión: ", ann[7]);


end;
