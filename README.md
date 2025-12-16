# ML Comment Classifier

API for sentiment and toxicity classification of comments. Uses XLM-RoBERTa for multilingual support.

## Features

- Sentiment detection (positive/neutral/negative)
- Toxicity detection
- Multilingual support
- Batch processing
- FastAPI with automatic documentation

## Tech Stack

- FastAPI, PyTorch, Transformers
- XLM-RoBERTa-base (270M parameters)
- Baseline: TF-IDF + Logistic Regression

## Quick Start

**Option 1: Use with pre-trained Hugging Face models** (no local models needed):
```bash
# API will automatically download models from Hugging Face Hub
docker-compose up --build
```

**Option 2: Train your own models** - create `data/clean_comments.csv` with columns:
- `text` - comment text
- `sentiment` - positive/neutral/negative
- `toxicity` - toxic/non-toxic

Then train:
```bash
python src/transformers_model/train_transformer.py
```

**Run API**:
```bash
python run_api.py
# or
start_api.bat
```

API available at `http://localhost:8000`
- Documentation: `http://localhost:8000/docs`

## Configuration

The API can use either:
1. **Local models** (if available in `models/` directory)
2. **Hugging Face Hub models** (downloaded automatically)

Configure via environment variables:
```bash
# Optional: Specify Hugging Face models
SENTIMENT_HF_MODEL=cardiffnlp/twitter-xlm-roberta-base-sentiment
TOXICITY_HF_MODEL=cardiffnlp/twitter-roberta-base-offensive
```

## API Usage

**Single comment**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is an amazing product!"}'
```

Response:
```json
{
  "text": "This is an amazing product!",
  "sentiment": "positive",
  "sentiment_confidence": 0.98,
  "toxicity": "non-toxic",
  "toxicity_confidence": 0.99
}
```

## Model Training

Configuration in `src/config.py`:
- Max sequence length: 128 tokens
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 3
- Early stopping: patience=2

## Metrics

Models are evaluated using:
- Accuracy
- F1 Score (macro)
- Precision / Recall

## Docker

```bash
docker-compose up --build
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

