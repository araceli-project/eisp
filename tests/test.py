IMG_NUM_SAMPLES = 100


def test_feature_extraction():
    from src.eisp.proxy_tasks import FeatureVectors
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    import os
    import numpy as np

    # Create a simple dataset
    data = torch.randn(
        IMG_NUM_SAMPLES, 3, 32, 32
    )  # IMG_NUM_SAMPLES samples of 3x32x32 images
    labels = torch.randint(0, 10, (IMG_NUM_SAMPLES,))  # Random labels for 10 classes
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=10)

    # Define simple feature extraction functions
    def mean_feature(x):
        return x.mean(dim=(1, 2, 3)).numpy().reshape(-1, 1)

    def std_feature(x):
        return x.std(dim=(1, 2, 3)).numpy().reshape(-1, 1)

    def plus_constant_feature(x, constant=5):
        return (x.mean(dim=(1, 2, 3)) + constant).numpy().reshape(-1, 1)

    proxy_features_functions = [mean_feature, std_feature, plus_constant_feature]
    proxy_features_names = ["mean", "std", "plus_constant"]
    proxy_features_function_arguments = [None, None, {"constant": 10}]

    # Run feature extraction
    featureVectors = FeatureVectors.extract(
        dataloader,
        proxy_features_functions,
        proxy_features_names,
        proxy_features_function_arguments,
        store_path="./test_features",
    )
    features = featureVectors.get_all_features()

    # Check if features are extracted correctly
    assert len(features) == 3
    assert "mean" in features
    assert "std" in features
    assert "plus_constant" in features
    assert features["mean"].shape[0] == IMG_NUM_SAMPLES
    assert features["std"].shape[0] == IMG_NUM_SAMPLES
    assert features["plus_constant"].shape[0] == IMG_NUM_SAMPLES

    # Check if the plus_constant feature is correctly computed
    assert np.array_equal(features["plus_constant"], features["mean"] + 10)

    # Check feature getters
    mean_feature_loaded = featureVectors.get_feature("mean")
    assert np.array_equal(mean_feature_loaded, features["mean"])

    # Check if features are saved correctly
    loaded_featureVectors = FeatureVectors.from_files("./test_features")
    loaded_features = loaded_featureVectors.get_all_features()
    assert np.array_equal(loaded_features["mean"], features["mean"])
    assert np.array_equal(loaded_features["std"], features["std"])

    # Clean up
    for name in proxy_features_names:
        os.remove(os.path.join("./test_features", f"{name}.npy"))
    os.rmdir("./test_features")


def test_feature_extraction_parallel():
    from src.eisp.proxy_tasks import FeatureVectors
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    import os
    import numpy as np

    # Create a simple dataset
    data = torch.randn(
        IMG_NUM_SAMPLES, 3, 32, 32
    )  # IMG_NUM_SAMPLES samples of 3x32x32 images
    labels = torch.randint(0, 10, (IMG_NUM_SAMPLES,))  # Random labels for 10 classes
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=10)

    # Define simple feature extraction functions
    def mean_feature(x):
        return x.mean(dim=(1, 2, 3)).numpy().reshape(-1, 1)

    def std_feature(x):
        return x.std(dim=(1, 2, 3)).numpy().reshape(-1, 1)

    def plus_constant_feature(x, constant=5):
        return (x.mean(dim=(1, 2, 3)) + constant).numpy().reshape(-1, 1)

    proxy_features_functions = [mean_feature, std_feature, plus_constant_feature]
    proxy_features_names = ["mean", "std", "plus_constant"]
    proxy_features_function_arguments = [None, None, {"constant": 10}]

    # Run feature extraction in parallel
    featureVectors = FeatureVectors.extract(
        dataloader,
        proxy_features_functions,
        proxy_features_names,
        proxy_features_function_arguments,
        parallel=True,
        store_path="./test_features_parallel",
    )
    features = featureVectors.get_all_features()

    # Check if features are extracted correctly
    assert len(features) == 3
    assert "mean" in features
    assert "std" in features
    assert "plus_constant" in features
    assert features["mean"].shape[0] == IMG_NUM_SAMPLES
    assert features["std"].shape[0] == IMG_NUM_SAMPLES
    assert features["plus_constant"].shape[0] == IMG_NUM_SAMPLES

    # Check if the plus_constant feature is correctly computed
    assert np.array_equal(features["plus_constant"], features["mean"] + 10)
    # Clean up
    for name in proxy_features_names:
        os.remove(os.path.join("./test_features_parallel", f"{name}.npy"))
    os.rmdir("./test_features_parallel")


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
