from typing import List, Dict, Optional
from pathlib import Path
from app.models.classifier_models import MultiTaskClassifier, SentimentClassifier


class ClassifierService:
    def __init__(self, sentiment_model_dir: Path, toxicity_model_dir: Path = None,
                 sentiment_hf_model: Optional[str] = None,
                 toxicity_hf_model: Optional[str] = None):
        self.sentiment_model_dir = sentiment_model_dir
        self.toxicity_model_dir = toxicity_model_dir
        self.sentiment_hf_model = sentiment_hf_model
        self.toxicity_hf_model = toxicity_hf_model
        self._classifier = None
        self._loaded = False

    def load_models(self) -> bool:
        try:
            # Check if both local directories exist
            sentiment_exists = self.sentiment_model_dir and self.sentiment_model_dir.exists()
            toxicity_exists = self.toxicity_model_dir and self.toxicity_model_dir.exists()

            if sentiment_exists and toxicity_exists:
                print("Loading models from local directories...")
                self._classifier = MultiTaskClassifier(
                    self.sentiment_model_dir, 
                    self.toxicity_model_dir
                )
            elif sentiment_exists or toxicity_exists or self.sentiment_hf_model or self.toxicity_hf_model:
                print("Loading models from Hugging Face Hub...")
                self._classifier = MultiTaskClassifier(
                    sentiment_model_dir=self.sentiment_model_dir if sentiment_exists else None,
                    toxicity_model_dir=self.toxicity_model_dir if toxicity_exists else None,
                    sentiment_hf_model=self.sentiment_hf_model,
                    toxicity_hf_model=self.toxicity_hf_model
                )
            else:
                print("No local models found, using default Hugging Face models...")
                self._classifier = MultiTaskClassifier()

            self._loaded = True
            print(f"✓ Models loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
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

