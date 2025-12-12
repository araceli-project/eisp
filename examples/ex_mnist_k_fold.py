from sklearn.metrics import balanced_accuracy_score


from eisp.ensemble import EnsembleKFold
from eisp.proxy_tasks import FeatureVectors
from eisp.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
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


print("Starting K-Fold Cross-Validation Training...")

ensemble_k_fold: EnsembleKFold = EnsembleKFold(train_features, train_labels, debug=True)

ensemble_k_fold.train_k_fold(
    k=5,
    model_type="xgboost",
    optimization_trials=5,
    optimization_direction="maximize",
    metric_function=lambda y_true, y_pred: balanced_accuracy_score(
        y_true, np.argmax(y_pred, axis=1)
    ),
    should_extract_shap=True,
)

print("K-Fold Training complete.")
print("Best validation metrics for each fold:", ensemble_k_fold.val_metric_k_fold)
print("Shap aggregated values for each fold:", ensemble_k_fold.shap_aggregated_k_fold)

mean_shap_values_over_folds = {}
for feature in ensemble_k_fold.shap_aggregated_k_fold.keys():
    mean_shap_values_over_folds[feature] = np.mean(
        ensemble_k_fold.shap_aggregated_k_fold[feature]
    )
print("Mean SHAP values over folds:", mean_shap_values_over_folds)

k_fold_feature_importance_save_path = "./data/mnist_vis/feature_importance_k_fold.png"
plot_feature_importance(
    mean_shap_values_over_folds, k_fold_feature_importance_save_path
)

print(
    f"Feature importance plot for K-Fold saved to {k_fold_feature_importance_save_path}"
)

print("Plotting confusion matrix for K-Fold")
confusion_matrix_k_fold_save_path = "./data/mnist_vis/confusion_matrix_k_fold.png"

# generate pred_labels and true_labels by concatenating predictions from each fold
all_pred_labels = []
for fold_pred in ensemble_k_fold.pred_labels_k_fold:
    all_pred_labels.append(np.argmax(fold_pred, axis=1))
all_pred_labels = np.concatenate(all_pred_labels, axis=0)

all_true_labels = np.concatenate(ensemble_k_fold.true_labels_k_fold, axis=0)

plot_confusion_matrix(
    true_labels=all_true_labels,
    pred_labels=all_pred_labels,
    class_names=[str(i) for i in range(10)],
    save_path=confusion_matrix_k_fold_save_path,
)

print("Example script for MNIST with K-Fold Cross-Validation completed successfully.")
