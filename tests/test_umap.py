def test_umap_plot():
    from src.eisp.visualization import plot_umap
    from src.eisp.proxy_tasks import FeatureVectors
    import numpy as np
    import os

    # Create dummy feature vectors
    num_samples = 100
    feature1 = np.random.rand(num_samples, 10)
    feature2 = np.random.rand(num_samples, 15)
    features_dict = {"feature1": feature1, "feature2": feature2}
    featureVectors = FeatureVectors(features_dict)

    # Create dummy labels
    labels = np.random.randint(0, 5, size=num_samples)

    # Generate UMAP plot
    save_path = "test_umap_plot.png"
    umap_results = plot_umap(
        featureVectors, labels, save_path=save_path, n_neighbors=10, min_dist=0.2
    )

    # Check if the UMAP results have the correct shape
    assert umap_results.shape == (num_samples, 2)

    # Check if the plot file is created
    assert os.path.exists(save_path)

    # Clean up
    os.remove(save_path)