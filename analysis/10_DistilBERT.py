import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import optuna
from datasets import Dataset
from helpers.split_group_stratified_and_join import split_group_stratified_and_join
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def check_cuda_availability():
    print("=" * 50)
    print("GPU AVAILABILITY CHECK")
    print("=" * 50)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(
            f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    print("=" * 50)
    return None


def load_data():
    print("Loading datasets...")
    df_train_lyrics = pd.read_csv("data/X_train_lyrics_dc.csv")
    df_test_lyrics = pd.read_csv("data/X_test_lyrics_dc.csv")
    train_texts = df_train_lyrics["full_lyrics"].tolist()
    test_texts = df_test_lyrics["full_lyrics"].tolist()

    df_train_meta = pd.read_csv("data/X_train_metadata_dc.csv")
    df_test_meta = pd.read_csv("data/X_test_metadata_dc.csv")
    train_labels_text = df_train_meta["dc_detailed"].tolist()
    test_labels_text = df_test_meta["dc_detailed"].tolist()

    train_artists = df_train_meta["track.s.firstartist.name"].reset_index(drop=True)

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels_text)
    test_labels = le.transform(test_labels_text)

    num_labels = len(le.classes_)
    print(f"Found {num_labels} unique classes: {le.classes_}")

    return (
        train_texts,
        test_texts,
        train_labels,
        test_labels,
        num_labels,
        train_artists,
    )


def create_and_tokenize_datasets(
    train_texts, test_texts, train_labels, test_labels, train_artists
):
    print("Creating Hugging Face datasets...")

    X_train = pd.DataFrame({"text": train_texts})
    labels_and_group = pd.DataFrame({"group": train_artists, "label": train_labels})

    X_train_split, X_val_split, y_train_split, y_val_split = (
        split_group_stratified_and_join(
            labels_and_group=labels_and_group,
            X=X_train,
            test_size=0.2,
            random_state=42,
        )
    )

    train_dataset = Dataset.from_dict(
        {"text": X_train_split["text"].tolist(), "label": y_train_split.tolist()}
    )
    val_dataset = Dataset.from_dict(
        {"text": X_val_split["text"].tolist(), "label": y_val_split.tolist()}
    )
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train, tokenized_val, tokenized_test, tokenizer


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="macro")
    kappa = cohen_kappa_score(labels, preds)
    return {"f1": f1, "kappa": kappa}


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [2, 3, 4]),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )

        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


def build_class_weights(train_dataset, num_labels):
    labels = np.array(train_dataset["label"])
    counts = np.bincount(labels, minlength=num_labels).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    weights = len(labels) / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(train_dataset, val_dataset, num_labels):
    """Initializes and trains the DistilBERT model."""
    print("Initializing model and trainer...")

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )

    class_weights = build_class_weights(train_dataset, num_labels)

    training_args = TrainingArguments(
        output_dir="models/distilbert",
        per_device_eval_batch_size=64,
        logging_dir="models/distilbert/logs",
        logging_steps=10,
        do_eval=True,
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting hyperparameter tuning...")

    os.makedirs("models/distilbert", exist_ok=True)
    db_path = "sqlite:///models/distillBERT/tune.sqlite3"

    study = optuna.create_study(
        study_name="distilbert_tuning",
        storage=db_path,
        direction="maximize",
        load_if_exists=True,
    )

    def objective(trial):
        params = optuna_hp_space(trial)
        for k, v in params.items():
            setattr(trainer.args, k, v)

        trainer.train()
        eval_metrics = trainer.evaluate()
        return eval_metrics["eval_f1"]

    study.optimize(objective, n_trials=20)

    print(f"\nBest Hyperparameters found:\n{study.best_params}")

    for k, v in study.best_params.items():
        setattr(trainer.args, k, v)

    print("\nRetraining with best hyperparameters...")
    trainer.train()

    print("Training finished.")
    return trainer


def evaluate_on_test_set(trainer, test_dataset):
    """Evaluate the trained model on the test set and return metrics, predictions, labels."""
    test_results = trainer.evaluate(test_dataset)
    pred_output = trainer.predict(test_dataset)
    test_preds = pred_output.predictions.argmax(-1)
    test_labels = pred_output.label_ids
    return test_results, test_preds, test_labels


def save_classification_artifacts(test_labels, test_preds, save_dir):
    """Save classification report and confusion matrix artifacts."""
    os.makedirs(save_dir, exist_ok=True)

    report_path = os.path.join(save_dir, "classification_report.txt")
    report_text = classification_report(test_labels, test_preds, digits=4)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(test_labels, test_preds, normalize="true")
    cm_path = os.path.join(save_dir, "confusion_matrix.csv")
    pd.DataFrame(cm).to_csv(cm_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="binary", xticks_rotation=70, colorbar=False, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=1200)
    plt.close(fig)


def save_test_results(scores, save_dir):
    """Save the final test scores to a txt file."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "evaluation.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Final Test Set Scores:\n")
        f.write("=" * 25 + "\n")
        for key, value in scores.items():
            f.write(f"{key}: {value}\n")
    print(f"Test scores saved to {filepath}")


if __name__ == "__main__":
    check_cuda_availability()
    (
        train_texts,
        test_texts,
        train_labels,
        test_labels,
        num_labels,
        train_artists,
    ) = load_data()
    print("Data successfully loaded and encoded")

    train_dataset, val_dataset, test_dataset, tokenizer = create_and_tokenize_datasets(
        train_texts, test_texts, train_labels, test_labels, train_artists
    )
    print("Datasets successfully created and tokenized")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    print("\nSample of tokenized training data:")
    print(train_dataset[0])

    trainer = train_model(train_dataset, val_dataset, num_labels)

    final_test_scores, test_preds, test_labels = evaluate_on_test_set(
        trainer, test_dataset
    )
    print("\nFinal Test Set Scores:")
    print(final_test_scores)

    print("\nSaving final model...")
    model_save_path = "models/distilbert"
    trainer.model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

    save_test_results(final_test_scores, model_save_path)
    save_classification_artifacts(test_labels, test_preds, model_save_path)
