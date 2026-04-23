from sklearn.externals.array_api_compat import torch
import os
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json


from lightgbm import LGBMClassifier, early_stopping
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from helpers.load_data import load_interpretable_classification_data
from helpers.split_group_stratified_and_join import (
    split_group_stratified_and_join,
    create_artist_separated_folds,
    plot_comparison_genre_distributions,
    plot_and_save_fold_label_prevalence,
)

NFOLDS = 4
PERMUTATION_IMPORTANCE_REPEATS = 10
RANDOM_STATE = 42
DEVICE = "cpu"


class OptunaExperiment:
    def __init__(self, modelname, mode="lightGBM", n_trials=100):
        if mode not in ["lightGBM", "glmnet"]:
            raise ValueError("Mode must be either 'lightGBM' or 'glmnet'.")
        self.mode = mode
        self.modelname = modelname
        self.n_trials = n_trials
        self.best_params = None
        self.model = None

    def _objective(self, trial, X_train, y_train):
        if self.mode == "lightGBM":
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.1, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 70),
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
        if self.mode == "glmnet":
            param = {
                "C": trial.suggest_float("C", 1e-5, 1e5, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            }

        scores = []

        for train_idx, val_idx in self.folds:
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pipeline = Pipeline(steps=[])
            if self.mode == "lightGBM":
                pipeline.steps.append(
                    (
                        "classifier",
                        LGBMClassifier(
                            njobs=-1,
                            class_weight="balanced",
                            device=DEVICE,
                            random_state=RANDOM_STATE,
                            **param,
                        ),
                    ),
                )
                pipeline.fit(
                    X_tr,
                    y_tr,
                    classifier__eval_set=[(X_val, y_val)],
                    classifier__eval_metric="multi_logloss",
                    classifier__callbacks=[
                        early_stopping(stopping_rounds=100, verbose=False)
                    ],
                )

            if self.mode == "glmnet":
                pipeline.steps.append(
                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            solver="saga",
                            max_iter=1000,
                            random_state=RANDOM_STATE,
                            **param,
                        ),
                    ),
                )
                pipeline.fit(
                    X_tr,
                    y_tr,
                )

            preds = pipeline.predict(X_val)
            score = f1_score(y_val, preds, average="macro")
            scores.append(score)

        mean_score = sum(scores) / len(scores)
        if hasattr(self, "scores_path"):
            with open(self.scores_path, "a", encoding="utf-8") as f:
                f.write(
                    f"trial={trial.number}\tmean_f1_macro={mean_score:.6f}\t"
                    f"fold_scores={[round(s, 6) for s in scores]}\t"
                    f"params={json.dumps(param, sort_keys=True)}\n"
                )
        return mean_score

    def _create_stratified_grouped_folds(self):
        labels_and_group = pd.DataFrame(
            {"group": self.artists.reset_index(drop=True), "label": self.y_train}
        )
        self.folds = create_artist_separated_folds(
            labels_and_group,
            self.X_train,
            n_splits=self.nfolds,
            random_state=RANDOM_STATE,
        )
        if not os.path.exists(f"models/{self.modelname}"):
            os.makedirs(f"models/{self.modelname}")
        plot_and_save_fold_label_prevalence(self.y_train, self.folds).savefig(
            f"models/{self.modelname}/fold_label_prevalence.png", dpi=1200
        )

    def tune(self, X_train, y_train, artists, nfolds=5):
        self.X_train = X_train
        self.y_train = y_train
        self.artists = artists
        self.nfolds = nfolds
        if os.path.exists(f"models/{self.modelname}/tune.sqlite3"):
            os.remove(f"models/{self.modelname}/tune.sqlite3")

        self._create_stratified_grouped_folds()

        if not os.path.exists(f"models/{self.modelname}"):
            os.makedirs(f"models/{self.modelname}")

        scores_dir = f"models/{self.modelname}/tuning"
        os.makedirs(scores_dir, exist_ok=True)
        self.scores_path = f"{scores_dir}/scores.txt"
        with open(self.scores_path, "w", encoding="utf-8") as f:
            f.write("trial\tmean_f1_macro\tfold_scores\tparams\n")

        if self.mode == "lightGBM":
            sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
            n_trials = self.n_trials
        if self.mode == "glmnet":
            # make sure fixed parameters are also in grid
            c_grid = np.sort(
                np.concatenate(
                    [
                        np.logspace(-3, 0, 10),  # 10^-3 to 10^0 (= 1.0)
                        np.logspace(0, 3, 11)[1:],  # 10^0 to 10^3, skip duplicate 1.0
                    ]
                )
            )
            l1_ratio_grid = np.sort(
                np.concatenate([np.linspace(0, 0.5, 10), np.linspace(0.5, 1, 11)[1:]])
            )
            search_space = {
                "C": c_grid,
                "l1_ratio": l1_ratio_grid,
            }
            sampler = optuna.samplers.GridSampler(search_space)
            n_trials = len(c_grid) * len(l1_ratio_grid)

        study = optuna.create_study(
            study_name=f"{self.modelname}",
            storage=f"sqlite:///models/{self.modelname}/tune.sqlite3",
            direction="maximize",
            sampler=sampler,
        )
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=n_trials,
        )
        self.best_params = study.best_trial.params
        self.best_params["random_state"] = RANDOM_STATE
        return study

    def retrain_best(self, X_train=None, y_train=None, artists=None, params=None):
        if params is not None:
            self.best_params = params
            if X_train is None or y_train is None or artists is None:
                raise ValueError(
                    "Provide X_train, y_train and artists if params is given."
                )
            self.X_train = X_train
            self.y_train = y_train
            self.artists = artists
        X_tr, X_val, y_tr, y_val = split_group_stratified_and_join(
            labels_and_group=pd.DataFrame(
                {"group": self.artists.reset_index(drop=True), "label": self.y_train}
            ),
            X=self.X_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        if not os.path.exists(f"models/{self.modelname}"):
            os.makedirs(f"models/{self.modelname}")
        plot_comparison_genre_distributions(y_tr, y_val).savefig(
            f"models/{self.modelname}/train_val_genre_distributions.png", dpi=1200
        )

        self.pipeline = Pipeline(steps=[])
        if self.mode == "lightGBM":
            self.pipeline.steps.append(
                (
                    "classifier",
                    LGBMClassifier(
                        njobs=-1,
                        class_weight="balanced",
                        device=DEVICE,
                        **self.best_params,
                    ),
                ),
            )
            self.pipeline.fit(
                X_tr,
                y_tr,
                classifier__eval_set=[(X_val, y_val)],
                classifier__eval_metric="multi_logloss",
                classifier__callbacks=[
                    early_stopping(stopping_rounds=100, verbose=False)
                ],
            )
        if self.mode == "glmnet":
            print(f"Retraining with best params: {self.best_params}")
            self.pipeline.steps.append(
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="saga",
                        max_iter=10000,
                        **self.best_params,
                    ),
                ),
            )
            self.pipeline.fit(
                X_tr,
                y_tr,
            )

        self.permutation_importance = permutation_importance(
            self.pipeline.named_steps["classifier"],
            X_val,
            y_val,
            scoring="f1_macro",
            n_repeats=PERMUTATION_IMPORTANCE_REPEATS,
            random_state=RANDOM_STATE,
        )
        self.n_samples_train = len(y_tr)

    def evaluate(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.n_samples_val = len(y_test)
        self.y_pred = self.pipeline.predict(X_test)
        f1 = f1_score(y_test, self.y_pred, average="macro")
        kappa = cohen_kappa_score(y_test, self.y_pred)
        report = classification_report(y_test, self.y_pred)
        self.classes = self.pipeline.named_steps["classifier"].classes_
        self.printout = (
            f"F1 Score (Macro): {f1:.4f}\n"
            f"Cohen's Kappa: {kappa:.4f}\n"
            f"Training Samples: {self.n_samples_train}\n"
            f"Testing Samples: {self.n_samples_val}\n"
            f"Classification Report:\n{report}\n"
            f"Best Params:\n{self.best_params}\n"
            f"Permutation Importance:\n{pd.Series(self.permutation_importance.importances_mean, index=X_test.columns).sort_values(ascending=False)}\n"
        )
        self.confusion = confusion_matrix(y_test, self.y_pred, normalize="true")

    def save_pipeline_and_evaluation(self):
        joblib.dump(self.pipeline, f"models/{self.modelname}/pipeline.joblib")
        pd.Series(self.y_pred, name="pred").to_csv(
            f"models/{self.modelname}/y_pred.csv", index=False
        )
        pd.DataFrame(
            {
                "mean_importance": self.permutation_importance.importances_mean,
                "std_importance": self.permutation_importance.importances_std,
                "feature": self.X_test.columns,
                "repeats": PERMUTATION_IMPORTANCE_REPEATS,
            }
        ).to_csv(f"models/{self.modelname}/permutation_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_idx = self.permutation_importance.importances_mean.argsort()[::-1][:30]
        ax.barh(
            X_test.columns[sorted_idx],
            self.permutation_importance.importances_mean[sorted_idx],
            xerr=self.permutation_importance.importances_std[sorted_idx],
            align="center",
        )
        ax.set_xlabel("Permutation Importance")
        ax.set_title("Top 30 Feature Importances")
        plt.tight_layout()
        plt.savefig(f"models/{self.modelname}/permutation_importance.png", dpi=1200)

        display = ConfusionMatrixDisplay(
            confusion_matrix=self.confusion, display_labels=self.classes
        )
        ax = plt.figure(figsize=(8, 8)).add_subplot(111)
        display.plot(cmap="binary", xticks_rotation=70, colorbar=False, ax=ax)

        plt.savefig(f"models/{self.modelname}/confusion_matrix.png")
        with open(f"models/{self.modelname}/evaluation.txt", "w") as f:
            f.write(self.printout)
        return None


if __name__ == "__main__":
    (
        y_train,
        y_test,
        _,
        _,
        _,
        _,
        _,
        _,
        X_train_expressions,
        X_test_expressions,
        _,
        _,
    ) = load_interpretable_classification_data()

    artists = pd.read_csv("data/X_train_metadata_dc.csv")["track.s.firstartist.name"]

    datasets = {
        "lr_tuned_expressions": (X_train_expressions, X_test_expressions, "glmnet"),
        "lgbm_tuned_expressions": (X_train_expressions, X_test_expressions, "lightGBM"),
    }

    for modelname, (X_train, X_test, mode) in datasets.items():
        experiment = OptunaExperiment(modelname=modelname, mode=mode, n_trials=100)
        experiment.tune(X_train, y_train, artists, nfolds=NFOLDS)
        experiment.retrain_best()
        experiment.evaluate(X_test, y_test)
        experiment.save_pipeline_and_evaluation()
