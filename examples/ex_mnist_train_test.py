from sklearn.metrics import balanced_accuracy_score


from eisp.ensemble import Ensemble
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
features: FeatureVectors = FeatureVectors.extract(
    train_loader,
    feature_functions,
    feature_names,
    store_path=train_feature_path,
)


labels = []
for _, target in train_loader:
    labels.append(target.numpy())
labels = np.concatenate(labels, axis=0)
# print features and labels shape
labels = np.array(labels)
for name in feature_names:
    print(f"Features shape: {features.get_all_features()[name].shape}")
print(f"Labels shape: {labels.shape}")

train_features, test_features, train_indices, test_indices = features.train_test_split(
    test_size=0.2, random_state=42
)

train_labels = labels[train_indices]
test_labels = labels[test_indices]

# Initialize and train ensemble
ensemble_model = Ensemble(train_features, train_labels)
ensemble_model.train(
    model_type="xgboost",
    optimization_trials=5,
    optimization_direction="maximize",
    metric_function=lambda y_true, y_pred: balanced_accuracy_score(
        y_true, np.argmax(y_pred, axis=1)
    ),
    should_extract_shap=True,
)

shap_values = ensemble_model.shap
shap_aggregated = ensemble_model.shap_aggregated

# Plot feature importance
feature_importance_save_path = "./data/mnist_vis/feature_importance.png"
plot_feature_importance(
    shap_aggregated,
    save_path=feature_importance_save_path,
)
print(f"Feature importance plot saved to {feature_importance_save_path}")

print({k: v.shape for k, v in shap_values.items()})
print({k: v for k, v in shap_aggregated.items()})

print("Ensemble training on MNIST completed successfully.")
print(f"Val metric: {ensemble_model.val_metric}")


confusion_matrix_save_path = "./data/mnist_vis/confusion_matrix.png"
plot_confusion_matrix(
    true_labels=ensemble_model.true_labels,
    pred_labels=np.argmax(ensemble_model.pred_labels, axis=1),
    class_names=[str(i) for i in range(10)],
    save_path=confusion_matrix_save_path,
)
print(f"Confusion matrix plot saved to {confusion_matrix_save_path}")

all_test_features = np.concatenate(
    [test_features.get_all_features()[name] for name in feature_names], axis=1
)

ensemble_model.test_xgboost(all_test_features, test_labels)
