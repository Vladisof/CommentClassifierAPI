from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_FILE = "clean_comments.csv"
DATA_PATH = DATA_DIR / DATA_FILE

BASELINE_DIR = MODEL_DIR / "baseline"
TRANSFORMER_DIR = MODEL_DIR / "transformers"
SENTIMENT_MODEL_DIR = MODEL_DIR / "xlmr_sentiment"
TOXICITY_MODEL_DIR = MODEL_DIR / "xlmr_toxicity"

BASELINE_CONFIG = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.9,
    "max_iter": 1000,
    "class_weight": "balanced",
}

TRANSFORMER_CONFIG = {
    "base_model": "xlm-roberta-base",
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "fp16": True,
}

DATA_SPLIT_CONFIG = {
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42,
}

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def print_config():
    """Print all configuration settings."""
    print("="*60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Device: {DEVICE}")

    print("\n" + "-"*60)
    print("Baseline Model Config:")
    for key, value in BASELINE_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n" + "-"*60)
    print("Transformer Model Config:")
    for key, value in TRANSFORMER_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n" + "-"*60)
    print("Data Split Config:")
    for key, value in DATA_SPLIT_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n" + "-"*60)
    print("API Config:")
    for key, value in API_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n" + "="*60)


if __name__ == "__main__":
    print_config()

