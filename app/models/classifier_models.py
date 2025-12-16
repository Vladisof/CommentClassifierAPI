import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import json
from typing import Dict, List, Optional


class SentimentClassifier:
    def __init__(self, model_dir: Path, device: Optional[str] = None, hf_model_name: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else None
        self.hf_model_name = hf_model_name
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self._load_model()

    def _load_model(self):
        # Try loading from local directory first
        if self.model_dir and self.model_dir.exists():
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
                self.model.to(self.device)
                self.model.eval()

                label_mapping_path = self.model_dir / "label_mapping.json"
                with open(label_mapping_path, 'r') as f:
                    mappings = json.load(f)
                    self.label_to_id = mappings['label_to_id']
                    self.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
                return
            except Exception as e:
                print(f"Failed to load from local directory: {e}")

        # Fallback to Hugging Face Hub or use a default multilingual model
        if self.hf_model_name:
            model_name = self.hf_model_name
        else:
            # Use a lightweight multilingual sentiment model as default
            model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

        print(f"Loading model from Hugging Face Hub: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Create default label mapping based on model config
        num_labels = self.model.config.num_labels
        if num_labels == 3:
            # Sentiment model (negative, neutral, positive)
            self.id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        elif num_labels == 2:
            # Binary classification (non-toxic, toxic)
            self.id_to_label = {0: 'non-toxic', 1: 'toxic'}
        else:
            # Generic labels
            self.id_to_label = {i: f'label_{i}' for i in range(num_labels)}

        self.label_to_id = {v: k for k, v in self.id_to_label.items()}

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
            probs = torch.softmax(outputs.logits, dim=1)

        pred_id = torch.argmax(probs, dim=1).item()
        pred_label = self.id_to_label[pred_id]
        confidence = probs[0][pred_id].item()

        result = {'label': pred_label, 'confidence': float(confidence)}

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
                probs = torch.softmax(outputs.logits, dim=1)

            pred_ids = torch.argmax(probs, dim=1).cpu().numpy()

            for j, pred_id in enumerate(pred_ids):
                results.append({
                    'label': self.id_to_label[pred_id],
                    'confidence': float(probs[j][pred_id].item()),
                    'probabilities': {
                        self.id_to_label[k]: float(probs[j][k].item())
                        for k in range(len(self.id_to_label))
                    }
                })

        return results


class MultiTaskClassifier:
    def __init__(self, sentiment_model_dir: Path = None, toxicity_model_dir: Path = None,
                 device: Optional[str] = None,
                 sentiment_hf_model: Optional[str] = None,
                 toxicity_hf_model: Optional[str] = None):
        self.sentiment_classifier = SentimentClassifier(
            sentiment_model_dir, device, sentiment_hf_model or "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )
        self.toxicity_classifier = SentimentClassifier(
            toxicity_model_dir, device, toxicity_hf_model or "cardiffnlp/twitter-roberta-base-offensive"
        )

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

        return [
            {
                'text': text,
                'sentiment': sentiment_results[i]['label'],
                'sentiment_confidence': sentiment_results[i]['confidence'],
                'sentiment_probabilities': sentiment_results[i]['probabilities'],
                'toxicity': toxicity_results[i]['label'],
                'toxicity_confidence': toxicity_results[i]['confidence'],
                'toxicity_probabilities': toxicity_results[i]['probabilities']
            }
            for i, text in enumerate(texts)
        ]

