import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
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

    # Encode labels to integers
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


def train_model(train_dataset, val_dataset, num_labels):
    """Initializes and trains the DistilBERT model."""
    print("Initializing model and trainer...")

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_labels
    )

    # hyperparameters for fine-tuning
    training_args = TrainingArguments(
        output_dir="models/distillBERT",  # Directory to save model checkpoints
        num_train_epochs=3,  # Standard number of epochs for fine-tuning
        per_device_train_batch_size=16,  # Batch size, reduce to 8 if you get memory errors
        per_device_eval_batch_size=64,  # Larger batch size for evaluation
        warmup_steps=500,  # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # Strength of weight decay
        logging_dir="models/distillBERT/logs",  # Directory for storing logs
        logging_steps=10,
        do_eval=True,
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Use validation split, not test set
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Training finished.")
    return trainer


def evaluate_on_test_set(trainer, test_dataset):
    """Evaluate the trained model on the test set and return metrics."""
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    return test_results


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

    # Evaluate on test set and get final scores
    final_test_scores = evaluate_on_test_set(trainer, test_dataset)
    print("\nFinal Test Set Scores:")
    print(final_test_scores)

    print("\nSaving final model...")
    model_save_path = "models/distilbert_sota_final"
    trainer.model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
