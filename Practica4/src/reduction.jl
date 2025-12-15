using MultivariateStats, Statistics, DataFrames

function pca(inputs::DataFrame; n_components::Int=2)
    X = Matrix(inputs)
    Xc = X .- mean(X, dims=1)

    pca_model = fit(PCA, Xc'; maxoutdim=n_components)

    X_pca = MultivariateStats.transform(pca_model, Xc')'

    return DataFrame(X_pca, :auto), pca_model
end
