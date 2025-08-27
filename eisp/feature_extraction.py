import os
import tqdm
import numpy as np
import unittest


def FeatureExtraction(
    dataloader: any,
    proxy_features_functions: list,
    proxy_features_names: list[str] = None,
    store_path: str = "./features",
):
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    if not proxy_features_names:
        proxy_features_names = [
            f"feature_{i}" for i in range(len(proxy_features_functions))
        ]
    if len(proxy_features_functions) != len(proxy_features_names):
        raise ValueError(
            "Length of proxy_features_functions and proxy_features_names must match."
        )

    all_features = {name: [] for name in proxy_features_names}

    for data in tqdm.tqdm(dataloader, desc="Extracting features"):
        inputs, _ = data  # Assuming dataloader returns (inputs, labels)

        for func, name in zip(proxy_features_functions, proxy_features_names):
            features = func(inputs)  # Extract features using the provided function
            all_features[name].append(features)

    # Concatenate and save features
    for name in proxy_features_names:
        all_features[name] = np.concatenate(all_features[name], axis=0)

    if store_path:
        for name in proxy_features_names:
            np.save(os.path.join(store_path, f"{name}.npy"), all_features[name])

    return all_features


class TestFeatureExtraction(unittest.TestCase):
    def test_feature_extraction(self):
        from torch.utils.data import DataLoader, TensorDataset
        import torch

        # Create a simple dataset
        data = torch.randn(100, 3, 32, 32)  # 100 samples of 3x32x32 images
        labels = torch.randint(0, 10, (100,))  # Random labels for 10 classes
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=10)

        # Define simple feature extraction functions
        def mean_feature(x):
            return x.mean(dim=(1, 2, 3)).numpy().reshape(-1, 1)

        def std_feature(x):
            return x.std(dim=(1, 2, 3)).numpy().reshape(-1, 1)

        proxy_features_functions = [mean_feature, std_feature]
        proxy_features_names = ["mean", "std"]

        # Run feature extraction
        features = FeatureExtraction(
            dataloader,
            proxy_features_functions,
            proxy_features_names,
            store_path="./test_features",
        )

        # Check if features are extracted correctly
        self.assertIn("mean", features)
        self.assertIn("std", features)
        self.assertEqual(features["mean"].shape[0], 100)
        self.assertEqual(features["std"].shape[0], 100)

        # Clean up
        for name in proxy_features_names:
            os.remove(os.path.join("./test_features", f"{name}.npy"))
        os.rmdir("./test_features")
