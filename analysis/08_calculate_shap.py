import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# renaming for interpreation
style_labels = pd.read_csv("data/style_labels.csv")
topic_labels = pd.read_csv("data/topic_labels.csv")
style_labels["id"] = style_labels["topic_num"].apply(lambda x: f"style_{x}")
topic_labels["id"] = topic_labels["topic_num"].apply(lambda x: f"topic_{x}")
feature_name_lookup_dict = {
    **style_labels.set_index("id")["style_labels"].to_dict(),
    **topic_labels.set_index("id")["topic_labels"].to_dict(),
}


with open(
    "models/classificator_selected_rf_tuned_dc/complete_experiment.pkl", "rb"
) as f:
    experiment = pickle.load(f)
model = experiment["model"]
preprocessing = model.named_steps["scaler"]
rf = model.named_steps["classifier"]

X_test = experiment["X_test"]
feature_names = [feature_name_lookup_dict.get(c, c) for c in X_test.columns]
y_test = experiment["y_test"]
scaled_X_test = preprocessing.transform(X_test)


def stratified_holdout_feature_sample(X, y, n_samples):
    _, X_sample, _, y_sample = train_test_split(
        X, y, test_size=n_samples, stratify=y, random_state=42
    )
    return X_sample


n_samples = 150
subset_X_test = stratified_holdout_feature_sample(
    scaled_X_test, y_test, n_samples=n_samples
)


explainer = shap.TreeExplainer(rf, subset_X_test)

shap_values = explainer(subset_X_test)
with open(
    f"models/classificator_selected_rf_tuned_dc/shap_values_{n_samples}-samples.pkl",
    "wb",
) as f:
    pickle.dump(shap_values, f)


with open(
    f"models/classificator_selected_rf_tuned_dc/shap_values_{n_samples}-samples.pkl",
    "rb",
) as f:
    shap_values = pickle.load(f)

shap_values.feature_names = feature_names

class_index_map = {i: cls for i, cls in enumerate(rf.classes_)}
print(class_index_map)
plt.figure(figsize=(6.5, 5.5))
shap.plots.beeswarm(shap_values[:, :, 9], max_display=20)
plt.savefig("reports/paper_ismir/figures/shap_beeswarm_rock.png", dpi=1200)
