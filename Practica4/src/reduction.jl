using MultivariateStats, Statistics, DataFrames, LinearAlgebra, Random

function pca(inputs::DataFrame; n_components::Int=2)
    X = Matrix(inputs)
    Xc = X .- mean(X, dims=1)

    pca_model = fit(PCA, Xc'; maxoutdim=n_components)

    X_pca = MultivariateStats.transform(pca_model, Xc')'

    return DataFrame(X_pca, :auto), pca_model
end





function lda(inputs::AbstractArray{<:Real,2}, labels::AbstractArray)

    X = Matrix{Float64}(inputs)
    classes = unique(labels)
    n_features = size(X, 2)

    global_mean = mean(X, dims=1)

    # matrices de dispersión
    S_W = zeros(n_features, n_features)
    S_B = zeros(n_features, n_features)

    for c in classes
        X_class = X[labels .== c, :]
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

    # ordenar por autovalor descendente
    idx = sortperm(eigvals, rev=true)
    eigvecs = eigvecs[:, idx]
    W = eigvecs[:, 1]

    # proyección de los datos en el nuevo subespacio
    X_proj = X * W

    return W, X_proj
end






function fastica(X::AbstractArray{<:Real,2}; tol, max_iter)
    n, m = size(X)

    X_centered = X .- mean(X, dims=2)

    # blanqueo (whitening)
    cov_matrix = cov(permutedims(X_centered))   # cov espera instancias en filas
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

            if abs(w' * w_prev) > 1 - tol
                break
            end
        end

        W[i, :] .= w'
    end

    S = W * X_white

    return S, W
end
