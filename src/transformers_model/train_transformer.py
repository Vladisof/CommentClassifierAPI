import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import json

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.preprocessing import clean_text
from src.transformers_model.dataset import CommentDataset, create_label_mappings
from src.utils.metrics import compute_classification_metrics

DATA_PATH = project_root / "data" / "clean_comments.csv"
MODEL_DIR = project_root / "models"
SENTIMENT_MODEL_DIR = MODEL_DIR / "xlmr_sentiment"
TOXICITY_MODEL_DIR = MODEL_DIR / "xlmr_toxicity"

BASE_MODEL = "xlm-roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
NUM_EPOCHS = 5
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42


def load_and_prepare_data(
    data_path: Path,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df['text_clean'] = df['text'].apply(lambda x: clean_text(str(x), lowercase=False))
    df = df[df['text_clean'].str.strip() != '']

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df['sentiment'] if 'sentiment' in df.columns else None
    )

    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state,
        stratify=train_val_df['sentiment'] if 'sentiment' in train_val_df.columns else None
    )

    return train_df, val_df, test_df


def compute_metrics_for_trainer(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.argmax(predictions, axis=1)
    metrics = compute_classification_metrics(
        labels, preds,
        y_proba=torch.softmax(torch.tensor(predictions), dim=1).numpy()
    )

    return metrics


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_column: str,
    output_dir: Path,
    model_name: str = BASE_MODEL,
    max_length: int = MAX_LENGTH,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    num_epochs: int = NUM_EPOCHS
) -> Tuple[Trainer, Dict[str, int], Dict[int, str]]:

    labels = train_df[label_column].tolist()
    label_to_id, id_to_label = create_label_mappings(labels)
    num_labels = len(label_to_id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = CommentDataset(
        texts=train_df['text_clean'].tolist(),
        labels=train_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        label_to_id=label_to_id
    )

    val_dataset = CommentDataset(
        texts=val_df['text_clean'].tolist(),
        labels=val_df[label_column].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
        label_to_id=label_to_id
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        push_to_hub=False,
        seed=RANDOM_STATE,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_trainer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    label_mapping_path = output_dir / "label_mapping.json"
    with open(label_mapping_path, 'w') as f:
        json.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f, indent=2)

    return trainer, label_to_id, id_to_label


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        train_df, val_df, test_df = load_and_prepare_data(DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if 'sentiment' in train_df.columns:
        train_model(
            train_df, val_df, 'sentiment', SENTIMENT_MODEL_DIR,
            BASE_MODEL, MAX_LENGTH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
        )

    if 'toxicity' in train_df.columns:
        train_model(
            train_df, val_df, 'toxicity', TOXICITY_MODEL_DIR,
            BASE_MODEL, MAX_LENGTH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
        )


if __name__ == "__main__":
    main()

