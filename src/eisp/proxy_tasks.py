import os
import tqdm
import numpy as np
from sklearn.decomposition import PCA
from typing import Callable
from collections.abc import Iterable
import joblib
from typing_extensions import Self


class FeatureVectors:
    def __init__(self, features: dict):
        self.features: dict[str, np.ndarray] = features
        self.pca_processed: bool = False
        if len(features) == 0:
            self.tuples_count = 0
            self.features_num = 0
        else:
            self.tuples_count: int = features.values().__iter__().__next__().shape[0]
            self.features_num: int = len(features)

    def extract(
        dataloader: Iterable[any],
        proxy_features_functions: list[Callable[[any], np.ndarray]],
        proxy_features_names: list[str] = None,
        proxy_features_function_arguments: list[dict] = None,
        store_path: str = None,
        parallel: bool = False,
    ) -> Self:
        if store_path and not os.path.exists(store_path):
            os.makedirs(store_path)

        if not proxy_features_functions or len(proxy_features_functions) == 0:
            raise ValueError("proxy_features_functions must be provided.")

        if not proxy_features_names:
            proxy_features_names = [
                f"feature_{i}" for i in range(len(proxy_features_functions))
            ]
        if len(proxy_features_functions) != len(proxy_features_names):
            raise ValueError(
                "Length of proxy_features_functions and proxy_features_names must match."
            )
        if not proxy_features_function_arguments:
            proxy_features_function_arguments = [None] * len(proxy_features_functions)
        if len(proxy_features_functions) != len(proxy_features_function_arguments):
            raise ValueError(
                "Length of proxy_features_functions and proxy_features_function_arguments must match."
            )

        all_features = {name: [] for name in proxy_features_names}

        if parallel:
            # Parallel feature extraction
            def extract_features_feature(function, args, dataloader):
                results = []
                for data in dataloader:
                    inputs, _ = data  # Assuming dataloader returns (inputs, labels)
                    result = function(inputs, **(args or {}))
                    results.append(result)
                return results

            with joblib.Parallel(n_jobs=-1) as parallel:
                results = parallel(
                    joblib.delayed(extract_features_feature)(function, args, dataloader)
                    for function, args in zip(
                        proxy_features_functions,
                        proxy_features_function_arguments,
                    )
                )
            for name, result in zip(proxy_features_names, results):
                all_features[name].extend(result)
        else:
            # Sequential feature extraction
            for function, name, args in zip(
                proxy_features_functions,
                proxy_features_names,
                proxy_features_function_arguments,
            ):
                if not callable(function):
                    raise ValueError(
                        f"Feature extraction function for {name} is not callable."
                    )
                for data in tqdm.tqdm(
                    dataloader, desc="Extracting features for " + name
                ):
                    inputs, _ = data  # Assuming dataloader returns (inputs, labels)
                    features = function(
                        inputs, **(args or {})
                    )  # Extract features using the provided function
                    all_features[name].append(features)

        # Concatenate and save features
        for name in proxy_features_names:
            all_features[name] = np.concatenate(all_features[name], axis=0)

        if store_path:
            for name in proxy_features_names:
                np.save(os.path.join(store_path, f"{name}.npy"), all_features[name])

        return FeatureVectors(all_features)

    def from_files(store_path: str) -> Self:
        if not os.path.exists(store_path):
            raise ValueError(f"Directory {store_path} does not exist.")

        features = {}
        for file in os.listdir(store_path):
            if file.endswith(".npy"):
                name = file[:-4]  # Remove .npy extension
                features[name] = np.load(os.path.join(store_path, file))

        return FeatureVectors(features)

    def get_feature(self, name: str) -> np.ndarray | None:
        return self.features.get(name, None)

    def get_all_features(self) -> dict[str, np.ndarray]:
        return self.features

    def apply_pca(self) -> Self:
        if self.pca_processed:
            return self

        pca_features = {}
        for name, feature in self.features.items():
            # Test all powers of 2 less than the feature dimension
            possible_components_num = []
            power = 2
            while power < min(feature.shape):
                possible_components_num.append(power)
                power *= 2

            for num_components in possible_components_num:
                pca_model = PCA(n_components=num_components).fit(feature)
                if pca_model.explained_variance_ratio_.sum() >= 0.8:
                    pca_features[name] = pca_model.transform(feature)
                    break
            if name not in pca_features:
                pca_features[name] = feature  # If no PCA applied, keep original

        pca_feature_vectors = FeatureVectors(pca_features)

        # Avoid reapplying PCA multiple times, if saved to disk,
        # this flag will be False when loaded again
        pca_feature_vectors.pca_processed = True
        return pca_feature_vectors

    def __len__(self) -> int:
        return self.features_num

    def __getitem__(self, name: str) -> np.ndarray | None:
        return self.get_feature(name)

    def get_tuples_count(self) -> int:
        return self.tuples_count

    def get_feature_names(self) -> list[str]:
        return list(self.features.keys())

    def train_test_split(
        self, test_size: float = 0.2, random_state: int = None
    ) -> tuple[Self, Self, np.ndarray, np.ndarray]:
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must be between 0 and 1.")

        np.random.seed(random_state)
        indices = np.arange(self.tuples_count)
        np.random.shuffle(indices)

        split_idx = int(self.tuples_count * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_features = {}
        test_features = {}
        for name, feature in self.features.items():
            train_features[name] = feature[train_indices]
            test_features[name] = feature[test_indices]

        return (
            FeatureVectors(train_features),
            FeatureVectors(test_features),
            train_indices,
            test_indices,
        )
