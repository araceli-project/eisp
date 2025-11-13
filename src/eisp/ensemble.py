import xgboost
from sklearn.metrics import balanced_accuracy_score
from eisp.proxy_tasks import FeatureVectors
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Callable


class Ensemble:
    def __init__(self, feature_vectors: FeatureVectors, labels: np.ndarray):
        self.feature_vectors: FeatureVectors = feature_vectors
        self.training_labels: np.ndarray = labels
        self.model: any = None
        self.best_val_metric: float = None
        self.test_metric: float = None

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
            study = optuna.create_study(
                direction=optimization_direction,
                study_name="xgboost_ensemble_optimization",
            )
            study.optimize(optimize, n_trials=optimization_trials)

            best_params = study.best_params
            params.update(best_params)

        dtrain = xgboost.DMatrix(X_train, label=y_train)
        dval = xgboost.DMatrix(X_val, label=y_val)
        dtest = xgboost.DMatrix(X_test, label=y_test)

        model = xgboost.train(params, dtrain, num_boost_round=num_boost_round)

        preds = model.predict(dval)
        val_metric_value = metric_function(y_val, preds)
        preds_test = model.predict(dtest)
        test_metric_value = metric_function(y_test, preds_test)

        self.model = model
        self.best_val_metric = val_metric_value
        self.test_metric = test_metric_value
        return model, val_metric_value, test_metric_value
