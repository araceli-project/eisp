from sklearn.metrics import balanced_accuracy_score

from eisp.ensemble import Ensemble, EnsembleCombinatorics
from eisp.proxy_tasks import FeatureVectors
from eisp.visualization import (
    plot_feature_importance,
    plot_confusion_matrix,
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

ensemble_model = Ensemble(train_features, train_labels, debug=True)

print("Starting training of ensemble model...")
ensemble_model.train(
    model_type="xgboost",
    optimization_trials=5,
    optimization_direction="maximize",
    metric_function=lambda y_true, y_pred: balanced_accuracy_score(
        y_true, np.argmax(y_pred, axis=1)
    ),
    should_extract_shap=True,
)

print("Training complete.")
print("Best hyperparameters:", ensemble_model.hyperparams)
print("Val metric:", ensemble_model.val_metric)

print("Starting training of ensemble combinatorics model...")

ensemble_combinatorics: EnsembleCombinatorics = EnsembleCombinatorics.from_ensemble(
    ensemble_model
)

ensemble_combinatorics.train_combinatorics(
    model_type="xgboost",
    metric_function=lambda y_true, y_pred: balanced_accuracy_score(
        y_true, np.argmax(y_pred, axis=1)
    ),
)

print("Combinatorics training complete.")

print(
    "Best validation metric per feature combination:",
    ensemble_combinatorics.best_val_metric,
)
print("Best feature combination:", ensemble_combinatorics.best_feature_combination)


print("Plotting confusion matrix")

confusion_matrix_save_path = "./data/mnist_vis/confusion_matrix_combinatorics.png"
plot_confusion_matrix(
    true_labels=ensemble_combinatorics.best_true_labels,
    pred_labels=np.argmax(ensemble_combinatorics.best_pred_labels, axis=1),
    class_names=[str(i) for i in range(10)],
    save_path=confusion_matrix_save_path,
)

print(f"Confusion matrix plot saved to {confusion_matrix_save_path}")

print("Saving combinatorics training data to disk...")

data_save_path = "./data/mnist_vis/ensemble_combinatorics_data.csv"

ensemble_combinatorics.save_training_data_to_disk(data_save_path)
