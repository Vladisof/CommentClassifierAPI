import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from typing import Tuple

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
MAX_FEATURES = 50000
NGRAM_RANGE = (1, 2)
MAX_ITER = 1000


def load_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['text_clean'] = df['text'].apply(lambda x: clean_text(str(x), lowercase=True))
    df = df[df['text_clean'].str.strip() != '']
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df['sentiment'] if 'sentiment' in df.columns else None
    )

    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state,
        stratify=train_val_df['sentiment'] if 'sentiment' in train_val_df.columns else None
    )

    return train_df, val_df, test_df


def train_tfidf_vectorizer(
    texts: pd.Series,
    max_features: int = MAX_FEATURES,
    ngram_range: Tuple[int, int] = NGRAM_RANGE
) -> TfidfVectorizer:
    """
    print(f"Training TF-IDF vectorizer (max_features={max_features}, ngram_range={ngram_range})...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.9,
        strip_accents='unicode',
        lowercase=True,
        analyzer='word',
        token_pattern=r'\w{1,}',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )

    vectorizer.fit(texts)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    return vectorizer


def train_classifier(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    task_name: str = "classification",
    max_iter: int = MAX_ITER
) -> LogisticRegression:
    """
    Train Logistic Regression classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        task_name: Name of the classification task
        max_iter: Maximum number of iterations

    Returns:
        Trained LogisticRegression model
    """
    print(f"\nTraining Logistic Regression for {task_name}...")

    n_classes = len(np.unique(y_train))
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution: {y_train.value_counts().to_dict()}")

    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        solver='lbfgs',
        multi_class='auto',
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    train_score = clf.score(X_train, y_train)
    print(f"Training accuracy: {train_score:.4f}")

    val_score = clf.score(X_val, y_val)
    print(f"Validation accuracy: {val_score:.4f}")

    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)

    target_names = [str(label) for label in clf.classes_]
    metrics = evaluate_model(y_val, y_pred, y_proba, target_names=target_names, verbose=True)

    return clf


def save_models(
    vectorizer: TfidfVectorizer,
    sentiment_model: LogisticRegression,
    toxicity_model: LogisticRegression = None
) -> None:
    print(f"\nSaving models to {MODEL_DIR}...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, TFIDF_MODEL_PATH)
    print(f"Saved TF-IDF vectorizer to {TFIDF_MODEL_PATH}")

    joblib.dump(sentiment_model, SENTIMENT_MODEL_PATH)
    print(f"Saved sentiment model to {SENTIMENT_MODEL_PATH}")

    if toxicity_model is not None:
        joblib.dump(toxicity_model, TOXICITY_MODEL_PATH)
        print(f"Saved toxicity model to {TOXICITY_MODEL_PATH}")

    print("All models saved successfully!")


def main():
    print("="*60)
    print("BASELINE MODEL TRAINING (TF-IDF + Logistic Regression)")
    print("="*60)

    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo run this script, you need to:")
        print("1. Prepare a CSV file named 'clean_comments.csv'")
        print("2. Place it in the 'data/' directory")
        print("3. The CSV should have columns: 'text', 'sentiment', 'toxicity'")
        print("\nExample format:")
        print("text,sentiment,toxicity")
        print("\"This is great!\",positive,non-toxic")
        print("\"I hate this\",negative,non-toxic")
        return

    df = preprocess_data(df)
    train_df, val_df, test_df = split_data(df)
    vectorizer = train_tfidf_vectorizer(train_df['text_clean'])

    print("\nTransforming texts to TF-IDF features...")
    X_train = vectorizer.transform(train_df['text_clean'])
    X_val = vectorizer.transform(val_df['text_clean'])
    X_test = vectorizer.transform(test_df['text_clean'])

    print(f"Feature matrix shape - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Train sentiment classifier
    if 'sentiment' in train_df.columns:
        sentiment_model = train_classifier(
            X_train,
            train_df['sentiment'],
            X_val,
            val_df['sentiment'],
            task_name="Sentiment Classification"
        )
    else:
        print("\nWarning: 'sentiment' column not found. Skipping sentiment training.")
        sentiment_model = None

    # Train toxicity classifier
    toxicity_model = None
    if 'toxicity' in train_df.columns:
        toxicity_model = train_classifier(
            X_train,
            train_df['toxicity'],
            X_val,
            val_df['toxicity'],
            task_name="Toxicity Classification"
        )
    else:
        print("\nWarning: 'toxicity' column not found. Skipping toxicity training.")

    # Save models
    if sentiment_model is not None:
        save_models(vectorizer, sentiment_model, toxicity_model)

    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nModels saved in: {MODEL_DIR}")
    print("\nNext steps:")
    print("1. Run evaluate_baseline.py to test on the test set")
    print("2. Start training the transformer model")


if __name__ == "__main__":
    main()

