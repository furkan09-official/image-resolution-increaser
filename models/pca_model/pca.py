from sklearn.decomposition import PCA

def apply_pca(images, n_components=100):
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(images)
    return pca_features, pca

# Example usage
pca_features, pca = apply_pca(images)