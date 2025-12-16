from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator


class CommentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('Text cannot be empty')
        if len(v.split()) > 1000:
            raise ValueError(f'Text too long (max 1000 words)')
        return v


class CommentBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)

    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('texts list cannot be empty')
        if len(v) > 100:
            raise ValueError(f'Batch too large (max 100 items)')

        validated = []
        for i, text in enumerate(v):
            text = text.strip()
            if not text:
                raise ValueError(f'Text at index {i} is empty')
            if len(text) > 5000:
                raise ValueError(f'Text at index {i} too long')
            validated.append(text)

        return validated


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    sentiment_confidence: float = Field(..., ge=0.0, le=1.0)
    sentiment_probabilities: Optional[Dict[str, float]] = None
    toxicity: Optional[str] = None
    toxicity_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    toxicity_probabilities: Optional[Dict[str, float]] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int = Field(..., ge=0)


class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    device: str
