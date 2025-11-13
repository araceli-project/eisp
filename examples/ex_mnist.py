from sklearn.metrics import balanced_accuracy_score
import shutil


from eisp.ensemble import Ensemble
from eisp.proxy_tasks import FeatureVectors
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
train_feature_path = "./mnist_train_features"
train_features = FeatureVectors.extract(
    train_loader,
    feature_functions,
    feature_names,
    store_path=train_feature_path,
)
train_labels = []
for _, target in train_loader:
    train_labels.append(target.numpy())
train_labels = np.concatenate(train_labels, axis=0)

# Initialize and train ensemble
ensemble_model = Ensemble(train_features, train_labels)
ensemble_model.train(
    model_type="xgboost",
    optimization_trials=5,
    optimization_direction="maximize",
    metric_function=lambda y_true, y_pred: balanced_accuracy_score(
        y_true, np.argmax(y_pred, axis=1)
    ),
)
print("Ensemble training on MNIST completed successfully.")
print(f"Best validation metric: {ensemble_model.best_val_metric}")
print(f"Test metric: {ensemble_model.test_metric}")
shutil.rmtree(train_feature_path)
shutil.rmtree("./data")
