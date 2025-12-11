from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import umap
import eisp.proxy_tasks as pt
import numpy as np
import os


def plot_tsne(
    features: pt.FeatureVectors,
    labels: np.ndarray = None,
    save_path: str = "tsne_plot.png",
    perplexity: int = 30,
    random_state: int = None,
):
    all_features = features.get_all_features()
    if len(all_features) == 0:
        raise ValueError("No features available for t-SNE plotting.")
    if labels is not None and len(labels) != features.tuples_count:
        raise ValueError("Length of labels must match the number of feature vectors.")
    if labels is None:
        labels = np.zeros(features.tuples_count)
    concatenated_features = np.concatenate(list(all_features.values()), axis=1)
    if concatenated_features.shape[1] < 2:
        raise ValueError("At least two feature dimensions are required for t-SNE.")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(concatenated_features)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, s=5, cmap="viridis")
    plt.title("t-SNE Visualization of Feature Vectors")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    if "/" in save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    return tsne_results


def plot_umap(
    features: pt.FeatureVectors,
    labels: np.ndarray = None,
    save_path: str = "umap_plot.png",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = None,
):
    all_features = features.get_all_features()
    if len(all_features) == 0:
        raise ValueError("No features available for UMAP plotting.")
    if labels is not None and len(labels) != features.tuples_count:
        raise ValueError("Length of labels must match the number of feature vectors.")
    if labels is None:
        labels = np.zeros(features.tuples_count)
    concatenated_features = np.concatenate(list(all_features.values()), axis=1)
    if concatenated_features.shape[1] < 2:
        raise ValueError("At least two feature dimensions are required for UMAP.")
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    umap_results = umap_model.fit_transform(concatenated_features)

    plt.figure(figsize=(8, 6))
    plt.scatter(umap_results[:, 0], umap_results[:, 1], c=labels, s=5, cmap="viridis")
    plt.title("UMAP Visualization of Feature Vectors")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    if "/" in save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    return umap_results


def plot_tsne_per_feature(
    features: pt.FeatureVectors,
    labels: np.ndarray = None,
    save_dir: str = "tsne_per_feature",
    perplexity: int = 30,
    random_state: int = None,
):
    all_features = features.get_all_features()
    if len(all_features) == 0:
        raise ValueError("No features available for t-SNE plotting.")
    if labels is not None and len(labels) != features.tuples_count:
        raise ValueError("Length of labels must match the number of feature vectors.")
    if labels is None:
        labels = np.zeros(features.tuples_count)

    os.makedirs(save_dir, exist_ok=True)

    for feature_name, feature_data in all_features.items():
        if feature_data.shape[1] < 2:
            continue  # Skip features with less than 2 dimensions
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        tsne_results = tsne.fit_transform(feature_data)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            tsne_results[:, 0], tsne_results[:, 1], c=labels, s=5, cmap="viridis"
        )
        plt.title(f"t-SNE Visualization of Feature: {feature_name}")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        save_path = os.path.join(save_dir, f"{feature_name}_tsne_plot.png")
        plt.savefig(save_path)
        plt.close()


def plot_umap_per_feature(
    features: pt.FeatureVectors,
    labels: np.ndarray = None,
    save_dir: str = "umap_per_feature",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = None,
):
    all_features = features.get_all_features()
    if len(all_features) == 0:
        raise ValueError("No features available for UMAP plotting.")
    if labels is not None and len(labels) != features.tuples_count:
        raise ValueError("Length of labels must match the number of feature vectors.")
    if labels is None:
        labels = np.zeros(features.tuples_count)

    os.makedirs(save_dir, exist_ok=True)

    for feature_name, feature_data in all_features.items():
        if feature_data.shape[1] < 2:
            continue  # Skip features with less than 2 dimensions
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
        )
        umap_results = umap_model.fit_transform(feature_data)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            umap_results[:, 0], umap_results[:, 1], c=labels, s=5, cmap="viridis"
        )
        plt.title(f"UMAP Visualization of Feature: {feature_name}")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        save_path = os.path.join(save_dir, f"{feature_name}_umap_plot.png")
        plt.savefig(save_path)
        plt.close()


def plot_feature_importance(
    shap_agg_values: dict[str, float],
    save_path: str = "feature_importance.png",
):
    if not shap_agg_values:
        raise ValueError("SHAP values are required for feature importance plotting.")

    feature_names = list(shap_agg_values.keys())
    importance_values = list(shap_agg_values.values())

    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance_values, color="skyblue")
    plt.ylabel("Mean Absolute SHAP Value")
    plt.xlabel("Feature Name")
    plt.title("Feature Importance based on SHAP Values")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: list[str],
    save_path: str = "confusion_matrix.png",
):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
