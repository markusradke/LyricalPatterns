import os
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
import matplotlib.pyplot as plt

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
    plot_fold_label_prevalence,
)

NFOLDS = 5
RANDOM_STATE = 42


class LightGBMOptunaExperiment:
    def __init__(self, modelname, n_trials=100):
        self.modelname = modelname
        self.n_trials = n_trials
        self.best_params = None
        self.model = None

    def _objective(self, trial, X_train, y_train):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 20, 50),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.1, 1.0),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
            "random_state": RANDOM_STATE,
        }

        scores = []

        for train_idx, val_idx in self.folds:
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = LGBMClassifier(class_weight="balanced", n_jobs=-1, **param)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
            )
            preds = model.predict(X_val)
            score = f1_score(y_val, preds, average="macro")
            scores.append(score)

        return sum(scores) / len(scores)

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
        plot_fold_label_prevalence(self.y_train, self.folds).savefig(
            f"models/{self.modelname}/fold_label_prevalence.png", dpi=1200
        )

    def tune(self, X_train, y_train, artists, nfolds=5):
        self.X_train = X_train
        self.y_train = y_train
        self.artists = artists
        self.nfolds = nfolds
        self._create_stratified_grouped_folds()

        if not os.path.exists(f"models/{self.modelname}"):
            os.makedirs(f"models/{self.modelname}")
        study = optuna.create_study(
            study_name=f"lightGBM_{self.modelname}",
            storage=f"sqlite:///models/{self.modelname}/tune.sqlite3",
            direction="maximize",
        )
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=self.n_trials,
        )
        self.best_params = study.best_trial.params
        self.best_params["random_state"] = RANDOM_STATE
        return study

    def retrain_best(self):
        X_tr, X_val, y_tr, y_val = split_group_stratified_and_join(
            labels_and_group=pd.DataFrame(
                {"group": self.artists.reset_index(drop=True), "label": self.y_train}
            ),
            X=self.X_train,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        plot_comparison_genre_distributions(y_tr, y_val).savefig(
            f"models/{self.modelname}/train_val_genre_distributions.png", dpi=1200
        )
        self.model = LGBMClassifier(
            class_weight="balanced", n_jobs=-1, **self.best_params
        )
        self.model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=100, verbose=10)],
        )
        self.n_samples_train = len(y_tr)

    def evaluate(self, X_test, y_test):
        self.n_samples_val = len(y_test)
        y_pred = self.model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        kappa = cohen_kappa_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        self.classes = self.model.classes_
        self.printout = (
            f"F1 Score (Macro): {f1:.4f}\n"
            f"Cohen's Kappa: {kappa:.4f}\n"
            f"Training Samples: {self.n_samples_train}\n"
            f"Testing Samples: {self.n_samples_val}\n"
            f"Classification Report:\n{report}\n"
            f"Best Params:\n{self.best_params}\n"
        )
        self.confusion = confusion_matrix(y_test, y_pred, normalize="true")

    def save_model_evaluation(self):
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
        X_train_topics,
        X_test_topics,
        X_train_styles,
        X_test_styles,
        X_train_combined,
        X_test_combined,
    ) = load_interpretable_classification_data()

    artists = pd.read_csv("data/X_train_metadata_dc.csv")["track.s.firstartist.name"]

    datasets = {
        "lgbm_topics": (X_train_topics, X_test_topics),
        "lgbm_styles": (X_train_styles, X_test_styles),
        "lgbm_combined": (X_train_combined, X_test_combined),
    }

    for modelname, (X_train, X_test) in datasets.items():
        experiment = LightGBMOptunaExperiment(modelname=modelname, n_trials=100)
        experiment.tune(X_train, y_train, artists, nfolds=NFOLDS)
        experiment.retrain_best()
        experiment.evaluate(X_test, y_test)
        experiment.save_model_evaluation()
