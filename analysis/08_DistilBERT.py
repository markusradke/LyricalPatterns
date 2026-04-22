import os
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import optuna
from datasets import Dataset
from sklearn.metrics import f1_score, cohen_kappa_score


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

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels_text)
    test_labels = le.transform(test_labels_text)

    num_labels = len(le.classes_)
    print(f"Found {num_labels} unique classes: {le.classes_}")

    return train_texts, test_texts, train_labels, test_labels, num_labels


def create_and_tokenize_datasets(train_texts, test_texts, train_labels, test_labels):
    print("Creating Hugging Face datasets...")
    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Create validation split from training data (80/20 split)
    train_val_split = tokenized_train.train_test_split(test_size=0.2, seed=42)
    tokenized_train = train_val_split["train"]
    tokenized_val = train_val_split["test"]

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


def train_model(train_dataset, val_dataset, num_labels):
    """Initializes and trains the DistilBERT model."""
    print("Initializing model and trainer...")

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )

    training_args = TrainingArguments(
        output_dir="models/distillBERT",
        per_device_eval_batch_size=64,
        logging_dir="models/distillBERT/logs",
        logging_steps=10,
        do_eval=True,
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting hyperparameter tuning...")

    os.makedirs("models/distilbert_sota_final", exist_ok=True)
    db_path = "sqlite:///models/distilbert/tune.sqlite3"

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
    """Evaluate the trained model on the test set and return metrics."""
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    return test_results


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
    train_texts, test_texts, train_labels, test_labels, num_labels = load_data()
    print("Data successfully loaded and encoded")

    train_dataset, val_dataset, test_dataset, tokenizer = create_and_tokenize_datasets(
        train_texts, test_texts, train_labels, test_labels
    )
    print("Datasets successfully created and tokenized")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    print("\nSample of tokenized training data:")
    print(train_dataset[0])

    trainer = train_model(train_dataset, val_dataset, num_labels)

    final_test_scores = evaluate_on_test_set(trainer, test_dataset)
    print("\nFinal Test Set Scores:")
    print(final_test_scores)

    print("\nSaving final model...")
    model_save_path = "models/distilbert_sota_final"
    trainer.model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

    save_test_results(final_test_scores, model_save_path)
