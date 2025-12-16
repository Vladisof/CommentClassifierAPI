import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional


class SentimentClassifier:
    def __init__(self, model_dir: str or Path, device: Optional[str] = None):
        self.model_dir = Path(model_dir)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self._load_model()

    def _load_model(self):
        print(f"Loading model from {self.model_dir}...")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

        label_mapping_path = self.model_dir / "label_mapping.json"
        with open(label_mapping_path, 'r') as f:
            mappings = json.load(f)
            self.label_to_id = mappings['label_to_id']
            self.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}

        print(f"Model loaded on {self.device}")
        print(f"Classes: {list(self.label_to_id.keys())}")

    def predict(self, text: str, return_probabilities: bool = True) -> Dict:
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        pred_id = torch.argmax(probs, dim=1).item()
        pred_label = self.id_to_label[pred_id]
        confidence = probs[0][pred_id].item()

        result = {
            'label': pred_label,
            'confidence': float(confidence)
        }

        if return_probabilities:
            result['probabilities'] = {
                self.id_to_label[i]: float(probs[0][i].item())
                for i in range(len(self.id_to_label))
            }

        return result

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)

            pred_ids = torch.argmax(probs, dim=1).cpu().numpy()

            for j, pred_id in enumerate(pred_ids):
                pred_label = self.id_to_label[pred_id]
                confidence = probs[j][pred_id].item()

                result = {
                    'label': pred_label,
                    'confidence': float(confidence),
                    'probabilities': {
                        self.id_to_label[k]: float(probs[j][k].item())
                        for k in range(len(self.id_to_label))
                    }
                }
                results.append(result)

        return results


class MultiTaskClassifier:
    def __init__(
        self,
        sentiment_model_dir: str or Path,
        toxicity_model_dir: str or Path,
        device: Optional[str] = None
    ):
        self.sentiment_classifier = SentimentClassifier(sentiment_model_dir, device)
        self.toxicity_classifier = SentimentClassifier(toxicity_model_dir, device)

    def predict(self, text: str) -> Dict:
        sentiment_result = self.sentiment_classifier.predict(text)
        toxicity_result = self.toxicity_classifier.predict(text)

        return {
            'text': text,
            'sentiment': sentiment_result['label'],
            'sentiment_confidence': sentiment_result['confidence'],
            'sentiment_probabilities': sentiment_result['probabilities'],
            'toxicity': toxicity_result['label'],
            'toxicity_confidence': toxicity_result['confidence'],
            'toxicity_probabilities': toxicity_result['probabilities']
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        sentiment_results = self.sentiment_classifier.predict_batch(texts, batch_size)
        toxicity_results = self.toxicity_classifier.predict_batch(texts, batch_size)

        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'sentiment': sentiment_results[i]['label'],
                'sentiment_confidence': sentiment_results[i]['confidence'],
                'sentiment_probabilities': sentiment_results[i]['probabilities'],
                'toxicity': toxicity_results[i]['label'],
                'toxicity_confidence': toxicity_results[i]['confidence'],
                'toxicity_probabilities': toxicity_results[i]['probabilities']
            }
            results.append(result)

        return results


def main():
    """Demo inference script."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "models"

    sentiment_model_dir = model_dir / "xlmr_sentiment"
    toxicity_model_dir = model_dir / "xlmr_toxicity"

    print("="*60)
    print("TRANSFORMER MODEL INFERENCE DEMO")
    print("="*60)

    # Check if models exist
    if not sentiment_model_dir.exists():
        print(f"\nError: Sentiment model not found at {sentiment_model_dir}")
        print("Please run train_transformer.py first.")
        return

    # Load classifier
    try:
        if toxicity_model_dir.exists():
            print("\nLoading multi-task classifier (sentiment + toxicity)...")
            classifier = MultiTaskClassifier(sentiment_model_dir, toxicity_model_dir)
        else:
            print("\nLoading sentiment classifier only...")
            classifier = SentimentClassifier(sentiment_model_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Example texts
    example_texts = [
        "This is absolutely amazing! I love it so much!",
        "This is terrible and I hate it.",
        "It's okay, nothing special.",
        "You are stupid and worthless!",
        "Great product, would recommend to everyone!",
        "Worst experience ever, complete waste of money."
    ]

    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)

    for text in example_texts:
        print(f"\nText: '{text}'")
        prediction = classifier.predict(text)

        if isinstance(classifier, MultiTaskClassifier):
            print(f"  Sentiment: {prediction['sentiment']} "
                  f"(confidence: {prediction['sentiment_confidence']:.3f})")
            print(f"  Toxicity: {prediction['toxicity']} "
                  f"(confidence: {prediction['toxicity_confidence']:.3f})")
        else:
            print(f"  Prediction: {prediction['label']} "
                  f"(confidence: {prediction['confidence']:.3f})")

    print("\n" + "="*60)
    print("INFERENCE DEMO COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()

