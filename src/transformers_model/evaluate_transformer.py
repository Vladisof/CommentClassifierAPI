import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import json
from typing import Dict, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.preprocessing import clean_text
from src.transformers_model.dataset import CommentDataset
from src.utils.metrics import evaluate_model

DATA_PATH = project_root / "data" / "clean_comments.csv"
MODEL_DIR = project_root / "models"
SENTIMENT_MODEL_DIR = MODEL_DIR / "xlmr_sentiment"
TOXICITY_MODEL_DIR = MODEL_DIR / "xlmr_toxicity"

TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42
BATCH_SIZE = 32
MAX_LENGTH = 128


def load_model_and_tokenizer(model_dir: Path) -> Tuple:
    print(f"Loading model from {model_dir}...")

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found at {model_dir}. "
            f"Please run train_transformer.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    label_mapping_path = model_dir / "label_mapping.json"
    with open(label_mapping_path, 'r') as f:
        mappings = json.load(f)
        label_to_id = mappings['label_to_id']
        id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print(f"Model loaded. Device: {device}")
    print(f"Label mapping: {label_to_id}")

    return model, tokenizer, label_to_id, id_to_label, device


def load_test_data() -> pd.DataFrame:
    """
    Load and preprocess test data.

    Returns:
        Test DataFrame
    """
    print(f"\nLoading data from {DATA_PATH}...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. "
            f"Please place your clean_comments.csv file in the /data/ directory."
        )

    df = pd.read_csv(DATA_PATH)

    # Preprocess
    df['text_clean'] = df['text'].apply(lambda x: clean_text(str(x), lowercase=False))
    df = df[df['text_clean'].str.strip() != '']

    # Split to get test set (using same random state as training)
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['sentiment'] if 'sentiment' in df.columns else None
    )

    print(f"Test set size: {len(test_df)} samples")

    return test_df


def predict_batch(
    model,
    tokenizer,
    texts: list,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on a batch of texts.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        texts: List of texts
        device: Device to run inference on
        batch_size: Batch size for inference
        max_length: Maximum sequence length

    Returns:
        Tuple of (predictions, probabilities)
    """
    model.eval()

    all_predictions = []
    all_probabilities = []

    # Create dataset without labels
    dataset = CommentDataset(
        texts=texts,
        labels=None,
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Get probabilities
            probs = torch.softmax(logits, dim=1)

            # Get predictions
            preds = torch.argmax(probs, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    return np.array(all_predictions), np.array(all_probabilities)


def evaluate_on_test_set(
    test_df: pd.DataFrame,
    model,
    tokenizer,
    label_to_id: Dict,
    id_to_label: Dict,
    device: torch.device,
    label_column: str
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Args:
        test_df: Test DataFrame
        model: Trained model
        tokenizer: Tokenizer
        label_to_id: Label to ID mapping
        id_to_label: ID to label mapping
        device: Device
        label_column: Name of label column

    Returns:
        Dictionary of metrics
    """
    print(f"\nEvaluating on test set...")

    # Get texts and true labels
    texts = test_df['text_clean'].tolist()
    true_labels_str = test_df[label_column].tolist()

    # Convert string labels to IDs
    true_labels = np.array([label_to_id[str(label)] for label in true_labels_str])

    # Make predictions
    print("Making predictions...")
    pred_labels, pred_probs = predict_batch(
        model, tokenizer, texts, device, batch_size=BATCH_SIZE
    )

    # Convert predictions back to string labels for display
    target_names = [id_to_label[i] for i in range(len(id_to_label))]

    # Evaluate
    metrics = evaluate_model(
        true_labels,
        pred_labels,
        pred_probs,
        target_names=target_names,
        verbose=True
    )

    return metrics


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("TRANSFORMER MODEL EVALUATION")
    print("="*60)

    # Load test data
    try:
        test_df = load_test_data()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    # Evaluate sentiment model
    if 'sentiment' in test_df.columns and SENTIMENT_MODEL_DIR.exists():
        print("\n" + "="*60)
        print("SENTIMENT CLASSIFICATION - TEST SET RESULTS")
        print("="*60)

        try:
            model, tokenizer, label_to_id, id_to_label, device = load_model_and_tokenizer(
                SENTIMENT_MODEL_DIR
            )

            sentiment_metrics = evaluate_on_test_set(
                test_df=test_df,
                model=model,
                tokenizer=tokenizer,
                label_to_id=label_to_id,
                id_to_label=id_to_label,
                device=device,
                label_column='sentiment'
            )

            print(f"\n✓ Sentiment evaluation completed!")

        except Exception as e:
            print(f"\n✗ Error evaluating sentiment model: {e}")

    # Evaluate toxicity model
    if 'toxicity' in test_df.columns and TOXICITY_MODEL_DIR.exists():
        print("\n" + "="*60)
        print("TOXICITY CLASSIFICATION - TEST SET RESULTS")
        print("="*60)

        try:
            model, tokenizer, label_to_id, id_to_label, device = load_model_and_tokenizer(
                TOXICITY_MODEL_DIR
            )

            toxicity_metrics = evaluate_on_test_set(
                test_df=test_df,
                model=model,
                tokenizer=tokenizer,
                label_to_id=label_to_id,
                id_to_label=id_to_label,
                device=device,
                label_column='toxicity'
            )

            print(f"\n✓ Toxicity evaluation completed!")

        except Exception as e:
            print(f"\n✗ Error evaluating toxicity model: {e}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()

