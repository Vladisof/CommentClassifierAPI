from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.services.classifier_service import ClassifierService
from app.schemas import (
    CommentRequest,
    CommentBatchRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse
)


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
SENTIMENT_MODEL_DIR = MODEL_DIR / "xlmr_sentiment"
TOXICITY_MODEL_DIR = MODEL_DIR / "xlmr_toxicity"

classifier_service = ClassifierService(SENTIMENT_MODEL_DIR, TOXICITY_MODEL_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading models...")
    success = classifier_service.load_models()
    if success:
        print(f"✓ Models loaded successfully on {classifier_service.device}")
    else:
        print("⚠ Failed to load models")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="Comment Classification API",
    description="Multilingual sentiment and toxicity classification",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/", response_model=dict)
async def root():
    return {
        "name": "Comment Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy" if classifier_service.is_loaded else "unhealthy",
        "message": "Models loaded" if classifier_service.is_loaded else "Models not loaded",
        "model_loaded": classifier_service.is_loaded,
        "device": classifier_service.device
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: CommentRequest):
    if not classifier_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        result = classifier_service.predict(request.text)
        return PredictionResponse(
            text=result['text'],
            sentiment=result['sentiment'],
            sentiment_confidence=result['sentiment_confidence'],
            sentiment_probabilities=result.get('sentiment_probabilities'),
            toxicity=result.get('toxicity'),
            toxicity_confidence=result.get('toxicity_confidence'),
            toxicity_probabilities=result.get('toxicity_probabilities')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: CommentBatchRequest):
    if not classifier_service.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        results = classifier_service.predict_batch(request.texts, batch_size=32)

        predictions = [
            PredictionResponse(
                text=result['text'],
                sentiment=result['sentiment'],
                sentiment_confidence=result['sentiment_confidence'],
                sentiment_probabilities=result.get('sentiment_probabilities'),
                toxicity=result.get('toxicity'),
                toxicity_confidence=result.get('toxicity_confidence'),
                toxicity_probabilities=result.get('toxicity_probabilities')
            )
            for result in results
        ]

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
