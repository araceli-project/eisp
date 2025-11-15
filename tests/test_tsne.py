def test_tsne_plot():
    from src.eisp.visualization import plot_tsne
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

    # Generate t-SNE plot
    save_path = "test_tsne_plot.png"
    tsne_results = plot_tsne(featureVectors, labels, save_path=save_path, perplexity=20)

    # Check if the t-SNE results have the correct shape
    assert tsne_results.shape == (num_samples, 2)

    # Check if the plot file is created
    assert os.path.exists(save_path)

    # Clean up
    os.remove(save_path)


def test_tsne_plot_no_features():
    from src.eisp.visualization import plot_tsne
    from src.eisp.proxy_tasks import FeatureVectors
    import numpy as np
    import pytest

    # Create empty feature vectors
    features_dict = {}
    featureVectors = FeatureVectors(features_dict)

    # Create dummy labels
    labels = np.random.randint(0, 5, size=100)

    # Expect ValueError when no features are available
    with pytest.raises(ValueError, match="No features available for t-SNE plotting."):
        plot_tsne(featureVectors, labels, save_path="dummy.png", perplexity=20)


def test_tsne_plot_label_length_mismatch():
    from src.eisp.visualization import plot_tsne
    from src.eisp.proxy_tasks import FeatureVectors
    import numpy as np
    import pytest

    # Create dummy feature vectors
    num_samples = 100
    feature1 = np.random.rand(num_samples, 10)
    features_dict = {"feature1": feature1}
    featureVectors = FeatureVectors(features_dict)

    # Create dummy labels with incorrect length
    labels = np.random.randint(0, 5, size=num_samples + 10)

    # Expect ValueError when label length does not match number of feature vectors
    with pytest.raises(
        ValueError, match="Length of labels must match the number of feature vectors."
    ):
        plot_tsne(featureVectors, labels, save_path="dummy.png", perplexity=20)


def test_tsne_plot_no_labels():
    from src.eisp.visualization import plot_tsne
    from src.eisp.proxy_tasks import FeatureVectors
    import numpy as np
    import os

    # Create dummy feature vectors
    num_samples = 100
    feature1 = np.random.rand(num_samples, 10)
    features_dict = {"feature1": feature1}
    featureVectors = FeatureVectors(features_dict)

    # Generate t-SNE plot without providing labels
    save_path = "test_tsne_plot_no_labels.png"
    tsne_results = plot_tsne(
        featureVectors, labels=None, save_path=save_path, perplexity=20
    )

    # Check if the t-SNE results have the correct shape
    assert tsne_results.shape == (num_samples, 2)

    # Check if the plot file is created
    assert os.path.exists(save_path)

    # Clean up
    os.remove(save_path)


def test_tsne_plot_insufficient_dimensions():
    from src.eisp.visualization import plot_tsne
    from src.eisp.proxy_tasks import FeatureVectors
    import numpy as np
    import pytest

    # Create dummy feature vectors with only one dimension
    num_samples = 100
    feature1 = np.random.rand(num_samples, 1)
    features_dict = {"feature1": feature1}
    featureVectors = FeatureVectors(features_dict)

    # Create dummy labels
    labels = np.random.randint(0, 5, size=num_samples)

    # Expect ValueError when there are insufficient feature dimensions for t-SNE
    with pytest.raises(
        ValueError, match="At least two feature dimensions are required for t-SNE."
    ):
        plot_tsne(featureVectors, labels, save_path="dummy.png", perplexity=20)
