import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from helpers.load_data import load_interpretable_classification_data

(
    y_train,
    y_test,
    _,
    _,
    _,
    _,
    _,
    _,
    _,
    _,
    X_train_combined,
    X_test_combined,
) = load_interpretable_classification_data()
pipeline = joblib.load("models/lgbm_combined/pipeline.joblib")
X_test_scaled = pipeline.named_steps["scaler"].transform(X_test_combined)

X_train_scaled = pipeline.named_steps["scaler"].transform(X_train_combined)

feature_names = X_train_combined.columns.tolist()

# Permutation Importance
perm_importance = permutation_importance(
    pipeline.named_steps["classifier"],
    X_test_scaled,
    y_test,
    n_repeats=10,
    random_state=42,
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
    "models/lgbm_combined/holdout_permutation_importance.csv", index=False
)

# SHAP Values
topic_labels = pd.read_csv("data/topic_labels.csv")
sentiment_labels = pd.read_csv("data/sentiment_labels.csv")
expressions_labels = pd.read_csv("data/expressions_labels.csv")
feature_name_lookup = {
    **{
        f"topic_{i}": topic_labels.loc[i, "topic_labels"]
        for i in range(len(topic_labels))
    },
    **{
        f"sentiment_{i}": sentiment_labels.loc[i, "sentiment_labels"]
        for i in range(len(sentiment_labels))
    },
    **{
        f"expressions_{i}": expressions_labels.loc[i, "expressions_labels"]
        for i in range(len(expressions_labels))
    },
}
feature_names = [feature_name_lookup.get(name, name) for name in feature_names]

classifier = pipeline.named_steps["classifier"]
explainer = shap.TreeExplainer(classifier)

# Focus on samples predicted as Hip Hop
y_pred = classifier.predict(X_test_scaled)
X_for_shap = pd.DataFrame(X_test_scaled, columns=feature_names)

shap_exp = explainer(X_for_shap)  # takes a while to compute
class_names = list(classifier.classes_)
hip_hop_idx = class_names.index("Hip Hop")

# Slice to 2D Explanation for the Hip Hop class
shap_exp_hip_hop = shap_exp[:, :, hip_hop_idx]

# Global feature impact (beeswarm) for Hip Hop prediction
ax = shap.plots.beeswarm(shap_exp_hip_hop, max_display=20, show=False)
fig = ax.get_figure()
ax.set_xlabel("SHAP Value (Impact on Model Output for Hip Hop)")
fig.tight_layout()

plt.savefig(
    "reports/paper_ismir/figures/shap_hip_hop.png", dpi=1200, bbox_inches="tight"
)
plt.close()
