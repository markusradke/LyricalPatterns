import re
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import permutation_importance

from helpers.load_data import load_interpretable_classification_data

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
    pipeline = joblib.load("models/lgbm_tuned_expressions/pipeline.joblib")

    X_train = X_train_expressions.copy()
    X_test = X_test_expressions.copy()

    feature_names = X_train_expressions.columns.tolist()
    print("Feature names:", feature_names)

    perm_importance = permutation_importance(
        pipeline.named_steps["classifier"],
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    df_perm_importance = pd.DataFrame(
        {
            "mean_importance": perm_importance.importances_mean,
            "std_importance": perm_importance.importances_std,
            "feature": feature_names,
        }
    ).sort_values(by="mean_importance", ascending=False)
    df_perm_importance["repeats"] = 10
    df_perm_importance.to_csv(
        "models/lgbm_tuned_expressions/holdout_permutation_importance.csv", index=False
    )

    # SHAP Values
    classifier = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)

    y_pred = classifier.predict(X_test)
    shap_exp = explainer(X_test)  # takes a while to compute
    class_names = list(classifier.classes_)
    custom_colors = LinearSegmentedColormap.from_list("custom", ["#B8B8B8", "#c40d20"])

    for class_idx, class_name in enumerate(class_names):
        shap_exp_class = shap_exp[:, :, class_idx]
        ax = shap.plots.beeswarm(
            shap_exp_class,
            max_display=20,
            show=False,
            color=custom_colors,
            plot_size=0.25,
        )
        fig = ax.get_figure()
        ax.set_xlabel(f"SHAP Value (Impact on Model Output for {class_name})", x=0.2)
        fig.tight_layout()
        class_name_formatted = re.sub(r"[,\s//&]", "", class_name).lower()
        plt.savefig(
            f"reports/paper_ismir/figures/shap_{class_name_formatted}.png",
            dpi=1200,
            bbox_inches="tight",
        )
        plt.close()
