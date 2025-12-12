import xgboost
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from eisp.proxy_tasks import FeatureVectors
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Callable
import shap
from typing_extensions import Self
from itertools import combinations
import pandas as pd


class Ensemble:
    def __init__(
        self, feature_vectors: FeatureVectors, labels: np.ndarray, debug=False
    ):
        self.feature_vectors: FeatureVectors = feature_vectors
        self.training_labels: np.ndarray = labels
        self.pred_labels: np.ndarray = None
        self.true_labels: np.ndarray = None
        self.model: any = None
        self.val_metric: float = None
        self.shap: dict[str, np.ndarray] = None
        self.shap_aggregated: dict[str, float] = None
        self.debug = debug
        self.hyperparams = None

    def train(
        self,
        model_type: str,
        hyperparams: dict = None,
        metric_function: Callable[
            [np.ndarray, np.ndarray], float
        ] = balanced_accuracy_score,
        optimization_trials: int = 0,
        optimization_direction: str = "maximize",
        num_boost_round: int = 100,
        should_extract_shap: bool = False,
    ):

        features = list(self.feature_vectors.get_all_features().values())
        X = np.concatenate(features, axis=1)
        y = self.training_labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )

        if model_type == "xgboost":
            return self.train_xgboost_model(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                optimization_trials,
                optimization_direction,
                metric_function,
                num_boost_round,
                should_extract_shap,
                hyperparams,
            )
        else:
            raise ValueError(f"Model type {model_type} not supported.")

    def train_xgboost_model(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        optimization_trials,
        optimization_direction,
        metric_function,
        num_boost_round,
        should_extract_shap: bool,
        params=None,
    ):
        if params is None:
            params = {
                "tree_method": "hist",
                "objective": (
                    "binary:logistic" if len(set(y_train)) == 2 else "multi:softprob"
                ),
                "num_class": len(set(y_train)),
                "eval_metric": "mlogloss",
                "seed": 42,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }

        def optimize(trial):
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
            params["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            )

            dtrain = xgboost.DMatrix(X_train, label=y_train)
            dval = xgboost.DMatrix(X_val, label=y_val)
            bst = xgboost.train(params, dtrain, num_boost_round=num_boost_round)
            preds = bst.predict(dval)
            return metric_function(y_val, preds)

        if optimization_trials > 0:
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(
                sampler=sampler,
                direction=optimization_direction,
                study_name="xgboost_ensemble_optimization",
            )
            study.optimize(optimize, n_trials=optimization_trials)

            best_params = study.best_params
            params.update(best_params)

        self.hyperparams = params
        dtrain = xgboost.DMatrix(X_train, label=y_train)
        dtest = xgboost.DMatrix(X_test, label=y_test)

        model = xgboost.train(params, dtrain, num_boost_round=num_boost_round)

        preds_test = model.predict(dtest)
        test_metric_value = metric_function(y_test, preds_test)
        self.pred_labels = preds_test
        self.true_labels = y_test

        self.model = model
        self.val_metric = test_metric_value

        if should_extract_shap:
            self.extract_aggregated_shap_values_per_feature_xb_boost(X_train)
        return model, test_metric_value

    def extract_aggregated_shap_values_per_feature_xb_boost(self, X: np.ndarray):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        feature_sizes = [
            feature.shape[1]
            for feature in self.feature_vectors.get_all_features().values()
        ]
        shap_per_feature = {}
        start_idx = 0
        for i, feature_name in enumerate(
            self.feature_vectors.get_all_features().keys()
        ):
            end_idx = start_idx + feature_sizes[i]
            shap_per_feature[feature_name] = np.sum(
                shap_values[:, start_idx:end_idx], axis=1
            )
            start_idx = end_idx
        self.shap = shap_per_feature
        self.shap_aggregated = {
            feature_name: np.abs(np.mean(shap_values))
            for feature_name, shap_values in shap_per_feature.items()
        }

        return shap_per_feature

    def test_xgboost(
        self, X_test: np.ndarray, y_test: np.ndarray, metric_function=None
    ):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        dtest = xgboost.DMatrix(X_test, label=y_test)
        preds_test = self.model.predict(dtest)
        if metric_function:
            test_metric_value = metric_function(y_test, preds_test)
        return test_metric_value, preds_test

    def predict_xgboost(self, X: np.ndarray):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        dmatrix = xgboost.DMatrix(X)
        preds = self.model.predict(dmatrix)
        return preds


class EnsembleKFold:
    def __init__(
        self, feature_vectors: FeatureVectors, labels: np.ndarray, debug=False
    ):
        self.feature_vectors: FeatureVectors = feature_vectors
        self.training_labels: np.ndarray = labels
        self.pred_labels_k_fold: list[np.ndarray] = None
        self.true_labels_k_fold: list[np.ndarray] = None
        self.shap_aggregated_k_fold: dict[str, list[float]] = None
        self.val_metric_k_fold: list[float] = None
        self.average_val_metric: float = None
        self.debug = debug
        self.hyperparams = None

    def train_k_fold(
        self,
        k: int,
        model_type: str,
        optimization_trials: int = 0,
        optimization_direction: str = "maximize",
        metric_function: Callable[
            [np.ndarray, np.ndarray], float
        ] = balanced_accuracy_score,
        num_boost_round: int = 100,
        should_extract_shap: bool = False,
    ):
        features = list(self.feature_vectors.get_all_features().values())
        X = np.concatenate(features, axis=1)
        y = self.training_labels

        if model_type == "xgboost":
            self.train_xgboost_model(
                k,
                X,
                y,
                optimization_trials,
                optimization_direction,
                metric_function,
                num_boost_round,
                should_extract_shap,
                self.hyperparams,
            )
        else:
            raise ValueError(f"Model type {model_type} not supported.")

    def train_xgboost_model(
        self,
        k: int,
        X,
        y,
        optimization_trials,
        optimization_direction,
        metric_function,
        num_boost_round,
        should_extract_shap: bool,
        params=None,
    ):
        if params is None:
            params = {
                "tree_method": "hist",
                "objective": (
                    "binary:logistic" if len(set(y)) == 2 else "multi:softprob"
                ),
                "num_class": len(set(y)),
                "eval_metric": "mlogloss",
                "seed": 42,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }

        def optimize(trial):
            params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3)
            params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
            params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
            params["colsample_bytree"] = trial.suggest_float(
                "colsample_bytree", 0.5, 1.0
            )
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            inst_val_metric_k_fold = []
            inst_pred_labels_k_fold = []
            inst_true_labels_k_fold = []
            inst_shap_aggregated_k_fold = {
                key: [] for key in self.feature_vectors.get_feature_names()
            }

            for i, (train_index, val_index) in enumerate(kf.split(X)):

                if self.debug:
                    print(f"Starting fold {i+1}/{k}")

                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                dtrain = xgboost.DMatrix(X_train, label=y_train)
                dval = xgboost.DMatrix(X_val, label=y_val)
                model = xgboost.train(params, dtrain, num_boost_round=num_boost_round)
                preds = model.predict(dval)
                inst_pred_labels_k_fold.append(preds)
                inst_true_labels_k_fold.append(y_val)
                val_metric_value = metric_function(y_val, preds)
                inst_val_metric_k_fold.append(val_metric_value)

                if self.debug:
                    print(f"Fold {i+1} Val Metric: {val_metric_value}")
                if should_extract_shap:
                    inst_shap_aggregated_k_fold = (
                        self.extract_aggregated_shap_values_per_feature_xb_boost(
                            X_train, model, inst_shap_aggregated_k_fold
                        )
                    )

            inst_average_val_metric = np.mean(inst_val_metric_k_fold)
            if (
                self.average_val_metric is None
                or inst_average_val_metric > self.average_val_metric
            ):
                print(
                    f"New best average val metric across folds: {inst_average_val_metric}"
                )
                self.average_val_metric = inst_average_val_metric
                self.val_metric_k_fold = inst_val_metric_k_fold
                self.pred_labels_k_fold = inst_pred_labels_k_fold
                self.true_labels_k_fold = inst_true_labels_k_fold
                self.shap_aggregated_k_fold = inst_shap_aggregated_k_fold
            return inst_average_val_metric

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            sampler=sampler,
            direction=optimization_direction,
            study_name="xgboost_ensemble_optimization",
        )
        study.optimize(optimize, n_trials=optimization_trials)

        best_params = study.best_params
        params.update(best_params)

    def extract_aggregated_shap_values_per_feature_xb_boost(
        self,
        X: np.ndarray,
        model,
        shap_aggregated_k_fold: dict[str, list[float]] = None,
    ):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        feature_sizes = [
            feature.shape[1]
            for feature in self.feature_vectors.get_all_features().values()
        ]
        shap_per_feature = {}
        start_idx = 0
        for i, feature_name in enumerate(
            self.feature_vectors.get_all_features().keys()
        ):
            end_idx = start_idx + feature_sizes[i]
            shap_per_feature[feature_name] = np.sum(
                shap_values[:, start_idx:end_idx], axis=1
            )
            start_idx = end_idx
        shap_aggregated = {
            feature_name: np.abs(np.mean(shap_values))
            for feature_name, shap_values in shap_per_feature.items()
        }
        shap_aggregated_k_fold = {
            feature_name: shap_aggregated_k_fold.get(feature_name, []) + [shap_value]
            for feature_name, shap_value in shap_aggregated.items()
        }
        return shap_aggregated_k_fold


class EnsembleCombinatorics:
    def __init__(
        self, feature_vectors: FeatureVectors, labels: np.ndarray, debug=False
    ):
        self.feature_vectors: FeatureVectors = feature_vectors
        self.training_labels: np.ndarray = labels
        self.best_pred_labels: np.ndarray = None
        self.best_true_labels: np.ndarray = None
        self.best_model: any = None
        self.best_val_metric: float = None
        self.best_feature_combination: list[str] = None
        self.debug = debug
        self.hyperparams = None
        self.training_data: list = None

    def from_ensemble(ensemble: Ensemble) -> Self:
        ensemble_comb = EnsembleCombinatorics(
            ensemble.feature_vectors, ensemble.training_labels, ensemble.debug
        )
        ensemble_comb.hyperparams = ensemble.hyperparams
        return ensemble_comb

    def train_combinatorics(
        self,
        model_type: str,
        metric_function: Callable[
            [np.ndarray, np.ndarray], float
        ] = balanced_accuracy_score,
        num_boost_round: int = 100,
    ):

        all_features_with_names = list(self.feature_vectors.get_all_features().items())

        self.training_data = []
        curr_val_metric = None

        for i in range(1, len(all_features_with_names) + 1):
            if self.debug:
                print(f"Evaluating feature combinations of size {i}")

            for feature_subset in combinations(all_features_with_names, i):
                feature_names = [feature[0] for feature in feature_subset]
                if self.debug:
                    print(f"Training with features: {feature_names}")

                X = np.concatenate([feature[1] for feature in feature_subset], axis=1)
                y = self.training_labels

                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                if model_type == "xgboost":
                    curr_val_metric = self.train_xgboost_model(
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        metric_function,
                        num_boost_round,
                        self.hyperparams,
                    )
                else:
                    raise ValueError(f"Model type {model_type} not supported.")

                self.training_data.append((feature_names, curr_val_metric))
                if self.debug:
                    print(
                        f"Feature combination: {feature_names}, Val Metric: {curr_val_metric}"
                    )
        self.training_data.sort(key=lambda x: x[1], reverse=True)
        self.best_feature_combination = self.training_data[0][0]

    def train_xgboost_model(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        metric_function,
        num_boost_round: int,
        params=None,
    ):
        if params is None:
            params = {
                "tree_method": "hist",
                "objective": (
                    "binary:logistic" if len(set(y_train)) == 2 else "multi:softprob"
                ),
                "num_class": len(set(y_train)),
                "eval_metric": "mlogloss",
                "seed": 42,
                "learning_rate": 0.1,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }

        dtrain = xgboost.DMatrix(X_train, label=y_train)
        dval = xgboost.DMatrix(X_val, label=y_val)
        model = xgboost.train(params, dtrain, num_boost_round=num_boost_round)
        preds = model.predict(dval)
        val_metric_value = metric_function(y_val, preds)

        if self.best_val_metric is None or val_metric_value > self.best_val_metric:
            self.best_val_metric = val_metric_value
            self.best_model = model
            self.best_pred_labels = preds
            self.best_true_labels = y_val

        return val_metric_value

    def test_xgboost(
        self, X_test: np.ndarray, y_test: np.ndarray, metric_function=None
    ):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        dtest = xgboost.DMatrix(X_test, label=y_test)
        preds_test = self.best_model.predict(dtest)
        if metric_function:
            test_metric_value = metric_function(y_test, preds_test)
        return test_metric_value, preds_test

    def predict_xgboost(self, X: np.ndarray):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        dmatrix = xgboost.DMatrix(X)
        preds = self.best_model.predict(dmatrix)
        return preds

    def save_training_data_to_disk(self, save_path: str):
        columns = ["feature_combination", "val_metric"]
        data = pd.DataFrame(self.training_data, columns=columns)
        data.to_csv(save_path, index=False)
