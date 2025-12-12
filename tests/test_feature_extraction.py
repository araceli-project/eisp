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


def test_feature_sep_train_test():
    from src.eisp.proxy_tasks import FeatureVectors
    from torch.utils.data import DataLoader, TensorDataset
    import torch
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

    proxy_features_functions = [mean_feature]
    proxy_features_names = ["mean"]

    # Run feature extraction
    featureVectors: FeatureVectors = FeatureVectors.extract(
        dataloader,
        proxy_features_functions,
        proxy_features_names,
    )

    train_features, test_features, train_indices, test_indices = (
        featureVectors.train_test_split(test_size=0.2, random_state=42)
    )

    # Check sizes
    assert train_features.get_feature("mean").shape[0] == 80
    assert test_features.get_feature("mean").shape[0] == 20

    # Check that indices are correct
    all_indices = np.concatenate([train_indices, test_indices])
    assert set(all_indices) == set(range(IMG_NUM_SAMPLES))
    assert len(set(train_indices).intersection(set(test_indices))) == 0

    # Check that features correspond to original
    original_features = featureVectors.get_feature("mean")
    assert np.array_equal(
        train_features.get_feature("mean"),
        original_features[train_indices],
    )
    assert np.array_equal(
        test_features.get_feature("mean"),
        original_features[test_indices],
    )
