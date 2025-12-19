using MultivariateStats, Statistics, DataFrames, LinearAlgebra, Random

function pca(inputs::DataFrame; n_components::Int=2)
    X = Matrix(inputs)
    Xc = X .- mean(X, dims=1)

    pca_model = MultivariateStats.fit(PCA, Xc'; maxoutdim=n_components)

    X_pca = MultivariateStats.transform(pca_model, Xc')'

    return DataFrame(X_pca, :auto), pca_model
end

function pca(inputs::DataFrame, threshold::Float64)
    X = Matrix(inputs)
    Xc = X .- mean(X, dims=1)

    pca_model = MultivariateStats.fit(PCA, Xc'; maxoutdim=size(X,2))

    coeffs = cumsum(principalvars(pca_model)) / sum(principalvars(pca_model))
    k = findfirst(>=(threshold), coeffs)
    X_pca = MultivariateStats.transform(pca_model, Xc')'[ :, 1:k]

    return DataFrame(X_pca, :auto), pca_model, k
end

function lda(dataset::Tuple{DataFrame, BitArray})
    inputs, labels = dataset;
    X = Matrix{Float64}(inputs)
    targets = [sum(findall(row)) for row in eachrow(labels)]
    classes = unique(targets)
    n_features = size(X, 2)

    global_mean = mean(X, dims=1)

    # matrices de dispersión
    S_W = zeros(n_features, n_features)
    S_B = zeros(n_features, n_features)

    for c in classes
        X_class = X[targets .== c, :]
        mean_class = mean(X_class, dims=1)

        # dispersión dentro de las clases
        for i in 1:size(X_class, 1)
            diff = X_class[i, :] .- mean_class
            S_W += diff' * diff
        end

        # dispersión entre clases
        n_c = size(X_class, 1)
        mean_diff = mean_class .- global_mean
        S_B += n_c * (mean_diff' * mean_diff)
    end

    # resolver autovalores
    eigvals, eigvecs = eigen(pinv(S_W) * S_B)

    # nos quedamos solo con la parte real de los autovalores complejos para que Julia pueda ordenarlos bien
    eigvals_real = real.(eigvals)
    idx = sortperm(eigvals_real, rev=true)
    eigvecs = eigvecs[:, idx]
    n_components = min(5, length(classes)-1)  # nunca más que n_classes-1
    W = eigvecs[:, 1:n_components]

    # proyección de los datos en el nuevo subespacio
    X_proj = X * W

    return X_proj, W
end

function fastica(dataset::Tuple{DataFrame, BitArray}; tol, max_iter)
    inputs, _ = dataset;
    X = Matrix{Float64}(inputs)
    n, m = size(X)

    X_centered = X .- mean(X, dims=2)

    # blanqueo (whitening)
    cov_matrix = cov(permutedims(X_centered))   # permutedims porque cov espera instancias en filas
    E, D, _ = svd(cov_matrix)
    D_inv = Diagonal(1.0 ./ sqrt.(D))
    whitening = E * D_inv * E'
    X_white = whitening * X_centered

    W = zeros(n, n)

    for i in 1:n
        w = randn(n)

        for _ in 1:max_iter
            w_prev = copy(w)
            w = (X_white * (tanh.(w' * X_white))') ./ m .- mean(1 .- (tanh.(w' * X_white)).^2) * w
            w ./= norm(w)

            if abs(dot(w, w_prev)) > 1 - tol
                break
            end
        end

        W[i, :] = w
    end

    S = W * X_white

    return S, W
end
