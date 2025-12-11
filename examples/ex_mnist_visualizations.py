from eisp.proxy_tasks import FeatureVectors
from eisp.visualization import (
    plot_tsne,
    plot_tsne_per_feature,
    plot_umap,
    plot_umap_per_feature,
)
import numpy as np

import torchvision
from torch.utils.data import DataLoader

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,)
        ),  # Mean and standard deviation for MNIST
    ]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# Define simple feature extraction functions
feature_names = ["image_itself", "image_mean", "image_std"]


def image_itself(x):
    return x.view(x.size(0), -1).numpy()


def image_mean(x):
    return x.view(x.size(0), -1).mean(dim=1).numpy().reshape(-1, 1)


def image_std(x):
    return x.view(x.size(0), -1).std(dim=1).numpy().reshape(-1, 1)


feature_functions = [image_itself, image_mean, image_std]

# Extract features for training set
train_feature_path = "./data/mnist_train_features"
train_features: FeatureVectors = FeatureVectors.extract(
    train_loader,
    feature_functions,
    feature_names,
    store_path=train_feature_path,
)
train_labels = []
for _, target in train_loader:
    train_labels.append(target.numpy())
train_labels = np.concatenate(train_labels, axis=0)

# Plot t-SNE of extracted features
tsne_save_path = "./data/mnist_vis/mnist_tsne_plot.png"
plot_tsne(
    train_features,
    labels=train_labels,
    save_path=tsne_save_path,
    perplexity=30,
)
print(f"t-SNE plot saved to {tsne_save_path}")

# Plot t-SNE of extracted features with PCA
train_features_pca = train_features.apply_pca()
tsne_save_path_pca = "./data/mnist_vis/mnist_tsne_plot_pca.png"
plot_tsne(
    train_features_pca,
    labels=train_labels,
    save_path=tsne_save_path_pca,
    perplexity=30,
)
print(f"t-SNE plot with PCA saved to {tsne_save_path_pca}")

# Plot t-SNE per feature with pca
tsne_per_feature_save_dir = "./data/mnist_vis/tsne_per_feature"
plot_tsne_per_feature(
    train_features_pca,
    labels=train_labels,
    save_dir=tsne_per_feature_save_dir,
    perplexity=30,
)
print(f"t-SNE per feature plots saved to {tsne_per_feature_save_dir}")

# Plot UMAP of extracted features with PCA
umap_save_path = "./data/mnist_vis/mnist_umap_plot_pca.png"
plot_umap(
    train_features_pca,
    labels=train_labels,
    save_path=umap_save_path,
    n_neighbors=15,
    min_dist=0.1,
)
print(f"UMAP plot saved to {umap_save_path}")

# Plot UMAP per feature with PCA
umap_per_feature_save_dir = "./data/mnist_vis/umap_per_feature"
plot_umap_per_feature(
    train_features_pca,
    labels=train_labels,
    save_dir=umap_per_feature_save_dir,
    n_neighbors=15,
    min_dist=0.1,
)
print(f"UMAP per feature plots saved to {umap_per_feature_save_dir}")
