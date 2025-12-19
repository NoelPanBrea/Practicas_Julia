using MultivariateStats, Statistics, DataFrames, LinearAlgebra, Random, Plots

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

# Visualización de los datos
function plot_pca_test(train_df::DataFrame, test_df::DataFrame, y_test)
    X_train = Matrix(train_df)
    X_test  = Matrix(test_df)

    # centrado usando solo train
    μ = mean(X_train, dims=1)
    X_train_c = X_train .- μ
    X_test_c  = X_test  .- μ

    # PCA entrenado en train
    pca_model = fit(PCA, X_train_c'; maxoutdim=2)

    # proyeccion de test
    X_test_pca = transform(pca_model, X_test_c')'

    scatter(
        X_test_pca[:, 1],
        X_test_pca[:, 2],
        group = y_test,
        legend = :outerright,
        title = "PCA – Proyección 2D (Test)",
        xlabel = "Componente 1",
        ylabel = "Componente 2",
        markersize = 3
    )
end


function plot_lda_test(train_df::DataFrame, test_df::DataFrame, y_train, y_test)
    # entrenar LDA en train
    X_train_lda, W = lda((train_df, y_train))

    # proyectar test usando W aprendido
    X_test = Matrix(test_df)
    X_test_lda = X_test * W[:, 1:2]

    scatter(
        X_test_lda[:, 1],
        X_test_lda[:, 2],
        group = y_test,
        legend = :outerright,
        title = "LDA – Proyección 2D (Test)",
        xlabel = "Componente 1",
        ylabel = "Componente 2",
        markersize = 3
    )
end


function plot_ica_test(train_df::DataFrame, test_df::DataFrame, y_test;
                       tol=1e-4, max_iter=500)

    # entrenar ICA en train
    S_train, W = fastica((train_df, falses(nrow(train_df))); 
                         tol=tol, max_iter=max_iter)

    X_train = Matrix(train_df)
    X_test  = Matrix(test_df)

    # centrado usando train
    μ = mean(X_train, dims=1)
    X_test_c = X_test .- μ

    cov_matrix = cov(permutedims(X_train .- μ))
    E, D, _ = svd(cov_matrix)
    whitening = E * Diagonal(1.0 ./ sqrt.(D)) * E'

    X_test_white = whitening * permutedims(X_test_c)

    # proyección ICA
    S_test = (W * X_test_white)[1:2, :]'

    scatter(
        S_test[:, 1],
        S_test[:, 2],
        group = y_test,
        legend = :outerright,
        title = "ICA – Proyección 2D (Test)",
        xlabel = "Componente 1",
        ylabel = "Componente 2",
        markersize = 3
    )
end

plot_pca_test(X_train_df, X_test_df, y_test)
plot_lda_test(X_train_df, X_test_df, y_train, y_test)
plot_ica_test(X_train_df, X_test_df, y_test)

