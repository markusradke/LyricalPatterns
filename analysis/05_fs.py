import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from helpers.load_data import load_interpretable_classification_data

RANDOM_STATE = 42


class FixedLinearModel:
    def __init__(self, modelname):
        self.modelname = modelname
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        l1_ratio=0.5,
                        penalty="elasticnet",
                        solver="saga",
                        class_weight="balanced",
                        verbose=1,
                        max_iter=50000,
                        random_state=42,
                    ),
                ),
            ]
        )

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_test):
        return self.pipeline.predict(X_test)

    def evaluate(self, X_test, y_test, top_coefficients=20):
        y_pred = self.pipeline.predict(X_test)
        self.n_features = self.pipeline.named_steps["classifier"].coef_.shape[1]
        self.classes = self.pipeline.classes_
        self.n_classes = len(self.classes)
        n_samples_train = len(self.y_train)
        n_samples_test = len(y_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        kappa = cohen_kappa_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        genres_top_coeffs = self._get_top_coefficients(top_n=top_coefficients)
        self.printout = (
            f"F1 Score: {f1:.4f}\n"
            f"Cohen's Kappa: {kappa:.4f}\n"
            f"Number of Features: {self.n_features}\n"
            f"Number of Classes: {self.n_classes}\n"
            f"Training Samples: {n_samples_train}\n"
            f"Testing Samples: {n_samples_test}\n"
            f"Classification Report:\n{report}\n"
            f"Top {top_coefficients} Coefficients per Genre:\n{genres_top_coeffs}"
        )

        self.confusion = confusion_matrix(y_test, y_pred, normalize="true")

    def _get_top_coefficients(self, top_n=20):
        top_coefficients = {
            self.pipeline.named_steps["classifier"].classes_[i]: sorted(
                zip(
                    range(self.n_features),
                    self.pipeline.named_steps["classifier"].coef_[i],
                ),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:top_n]
            for i in range(self.n_classes)
        }
        printout = "Top Coefficients per Genre:\n"
        for genre, coeffs in top_coefficients.items():
            printout += f"\nGenre: {genre}\n"
            for idx, coeff in coeffs:
                printout += f"Feature Index: {idx}, Coefficient: {coeff:.4f}\n"
        return printout

    def save_model_evaluation(self):
        assert hasattr(self, "printout") and hasattr(
            self, "confusion"
        ), "Evaluate the model before saving results."
        display = ConfusionMatrixDisplay(
            confusion_matrix=self.confusion, display_labels=self.classes
        )
        ax = plt.figure(figsize=(8, 8)).add_subplot(111)
        display.plot(cmap="binary", xticks_rotation=70, colorbar=False, ax=ax)
        plt.title(f"Confusion Matrix for {self.modelname}")

        if not os.path.exists(f"models/{self.modelname}"):
            os.makedirs(f"models/{self.modelname}")
        plt.savefig(f"models/{self.modelname}/confusion_matrix.png")
        with open(f"models/{self.modelname}/evaluation.txt", "w") as f:
            f.write(self.printout)
        return None


if __name__ == "__main__":
    print("Loading data...")
    (
        y_train,
        y_test,
        X_train_fs,
        X_test_fs,
        X_train_topics,
        X_test_topics,
        X_train_sentiments,
        X_test_sentiments,
        X_train_expressions,
        X_test_expressions,
        X_train_combined,
        X_test_combined,
    ) = load_interpretable_classification_data()

    print("Data successfully loaded.\nTraining linear models...")
    models = {
        # "lr_topics": (X_train_topics, X_test_topics),
        # "lr_sentiments": (X_train_sentiments, X_test_sentiments),
        # "lr_expressions": (X_train_expressions, X_test_expressions),
        # "lr_combined": (X_train_combined, X_test_combined),
        "lr_fs": (X_train_fs, X_test_fs),
    }
    for modelname, (X_train, X_test) in models.items():
        print(f"\nTraining {modelname} model...")
        model = FixedLinearModel(modelname)
        model.fit(X_train, y_train)
        print(f"Evaluating {modelname} model...")
        model.evaluate(X_test, y_test)
        print(model.printout)
        model.save_model_evaluation()
    print("All models trained and evaluated.")
