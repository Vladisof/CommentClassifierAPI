import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, Dict

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.preprocessing import clean_text
from src.utils.metrics import evaluate_model, print_metrics_summary

DATA_PATH = project_root / "data" / "clean_comments.csv"
MODEL_DIR = project_root / "models"
TFIDF_MODEL_PATH = MODEL_DIR / "baseline_tfidf.pkl"
SENTIMENT_MODEL_PATH = MODEL_DIR / "baseline_sentiment_logreg.pkl"
TOXICITY_MODEL_PATH = MODEL_DIR / "baseline_toxicity_logreg.pkl"

TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42


def load_models() -> Tuple:
    print("Loading trained models...")

    if not TFIDF_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found at {TFIDF_MODEL_PATH}. "
            f"Please run train_baseline.py first."
        )

    if not SENTIMENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Sentiment model not found at {SENTIMENT_MODEL_PATH}. "
            f"Please run train_baseline.py first."
        )

    vectorizer = joblib.load(TFIDF_MODEL_PATH)
    print(f"Loaded TF-IDF vectorizer from {TFIDF_MODEL_PATH}")

    sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
    print(f"Loaded sentiment model from {SENTIMENT_MODEL_PATH}")

    toxicity_model = None
    if TOXICITY_MODEL_PATH.exists():
        toxicity_model = joblib.load(TOXICITY_MODEL_PATH)
        print(f"Loaded toxicity model from {TOXICITY_MODEL_PATH}")

    return vectorizer, sentiment_model, toxicity_model


def load_test_data() -> pd.DataFrame:
    print(f"\nLoading data from {DATA_PATH}...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found at {DATA_PATH}. "
            f"Please place your clean_comments.csv file in the /data/ directory."
        )

    df = pd.read_csv(DATA_PATH)

    # Preprocess
    df['text_clean'] = df['text'].apply(lambda x: clean_text(str(x), lowercase=True))
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


def evaluate_on_test_set(
    test_df: pd.DataFrame,
    vectorizer,
    sentiment_model,
    toxicity_model=None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate models on test set.

    Args:
        test_df: Test DataFrame
        vectorizer: Trained TF-IDF vectorizer
        sentiment_model: Trained sentiment classifier
        toxicity_model: Trained toxicity classifier (optional)

    Returns:
        Dictionary containing metrics for each task
    """
    results = {}

    # Transform texts to TF-IDF features
    print("\nTransforming test texts to TF-IDF features...")
    X_test = vectorizer.transform(test_df['text_clean'])
    print(f"Test feature matrix shape: {X_test.shape}")

    # Evaluate sentiment model
    if 'sentiment' in test_df.columns:
        print("\n" + "="*60)
        print("SENTIMENT CLASSIFICATION - TEST SET RESULTS")
        print("="*60)

        y_test = test_df['sentiment']
        y_pred = sentiment_model.predict(X_test)
        y_proba = sentiment_model.predict_proba(X_test)

        target_names = [str(label) for label in sentiment_model.classes_]
        metrics = evaluate_model(y_test, y_pred, y_proba, target_names=target_names, verbose=True)

        results['sentiment'] = metrics

    # Evaluate toxicity model
    if toxicity_model is not None and 'toxicity' in test_df.columns:
        print("\n" + "="*60)
        print("TOXICITY CLASSIFICATION - TEST SET RESULTS")
        print("="*60)

        y_test = test_df['toxicity']
        y_pred = toxicity_model.predict(X_test)
        y_proba = toxicity_model.predict_proba(X_test)

        target_names = [str(label) for label in toxicity_model.classes_]
        metrics = evaluate_model(y_test, y_pred, y_proba, target_names=target_names, verbose=True)

        results['toxicity'] = metrics

    return results


def predict_single_comment(
    text: str,
    vectorizer,
    sentiment_model,
    toxicity_model=None
) -> Dict:
    """
    Make predictions on a single comment.

    Args:
        text: Input comment text
        vectorizer: Trained TF-IDF vectorizer
        sentiment_model: Trained sentiment classifier
        toxicity_model: Trained toxicity classifier (optional)

    Returns:
        Dictionary with predictions and probabilities
    """
    # Clean text
    text_clean = clean_text(text, lowercase=True)

    # Transform to TF-IDF features
    X = vectorizer.transform([text_clean])

    # Predict sentiment
    sentiment_pred = sentiment_model.predict(X)[0]
    sentiment_proba = sentiment_model.predict_proba(X)[0]
    sentiment_confidence = float(max(sentiment_proba))

    result = {
        'text': text,
        'text_clean': text_clean,
        'sentiment': str(sentiment_pred),
        'sentiment_confidence': sentiment_confidence,
        'sentiment_probabilities': {
            str(label): float(prob)
            for label, prob in zip(sentiment_model.classes_, sentiment_proba)
        }
    }

    # Predict toxicity if model available
    if toxicity_model is not None:
        toxicity_pred = toxicity_model.predict(X)[0]
        toxicity_proba = toxicity_model.predict_proba(X)[0]
        toxicity_confidence = float(max(toxicity_proba))

        result.update({
            'toxicity': str(toxicity_pred),
            'toxicity_confidence': toxicity_confidence,
            'toxicity_probabilities': {
                str(label): float(prob)
                for label, prob in zip(toxicity_model.classes_, toxicity_proba)
            }
        })

    return result


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("BASELINE MODEL EVALUATION")
    print("="*60)

    # Load models
    try:
        vectorizer, sentiment_model, toxicity_model = load_models()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    # Load test data
    try:
        test_df = load_test_data()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    # Evaluate on test set
    results = evaluate_on_test_set(test_df, vectorizer, sentiment_model, toxicity_model)

    # Demo: Predict on example comments
    print("\n" + "="*60)
    print("DEMO: PREDICTIONS ON EXAMPLE COMMENTS")
    print("="*60)

    example_comments = [
        "This is absolutely amazing! I love it so much!",
        "This is terrible and I hate it.",
        "It's okay, nothing special.",
        "You are stupid and worthless!"
    ]

    for comment in example_comments:
        print(f"\nComment: '{comment}'")
        prediction = predict_single_comment(comment, vectorizer, sentiment_model, toxicity_model)
        print(f"  Sentiment: {prediction['sentiment']} (confidence: {prediction['sentiment_confidence']:.3f})")
        if 'toxicity' in prediction:
            print(f"  Toxicity: {prediction['toxicity']} (confidence: {prediction['toxicity_confidence']:.3f})")

    print("\n" + "="*60)
    print("EVALUATION COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()

