from typing import List, Dict
from pathlib import Path
from app.models.classifier_models import MultiTaskClassifier, SentimentClassifier


class ClassifierService:
    def __init__(self, sentiment_model_dir: Path, toxicity_model_dir: Path = None):
        self.sentiment_model_dir = sentiment_model_dir
        self.toxicity_model_dir = toxicity_model_dir
        self._classifier = None
        self._loaded = False

    def load_models(self) -> bool:
        try:
            if self.toxicity_model_dir and self.toxicity_model_dir.exists():
                self._classifier = MultiTaskClassifier(
                    self.sentiment_model_dir, 
                    self.toxicity_model_dir
                )
            else:
                self._classifier = SentimentClassifier(self.sentiment_model_dir)
            
            self._loaded = True
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def predict(self, text: str) -> Dict:
        if not self._loaded:
            raise RuntimeError("Models not loaded")
        return self._classifier.predict(text)

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        if not self._loaded:
            raise RuntimeError("Models not loaded")
        return self._classifier.predict_batch(texts, batch_size)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        if self._loaded:
            return str(self._classifier.sentiment_classifier.device)
        return "unknown"

