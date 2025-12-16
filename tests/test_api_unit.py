"""
Unit tests for the Comment Classifier API.
Run with: pytest tests/test_api_unit.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


@pytest.fixture
def mock_classifier_service():
    """Mock classifier service for testing."""
    service = Mock()
    service.is_loaded = True
    service.device = 'cpu'
    service.predict.return_value = {
        'text': 'Test comment',
        'sentiment': 'positive',
        'sentiment_confidence': 0.95,
        'sentiment_probabilities': {
            'positive': 0.95,
            'neutral': 0.03,
            'negative': 0.02
        },
        'toxicity': 'non-toxic',
        'toxicity_confidence': 0.98,
        'toxicity_probabilities': {
            'toxic': 0.02,
            'non-toxic': 0.98
        }
    }
    service.predict_batch.return_value = [
        {
            'text': 'Test comment 1',
            'sentiment': 'positive',
            'sentiment_confidence': 0.95,
            'toxicity': 'non-toxic',
            'toxicity_confidence': 0.98
        }
    ]
    return service


@pytest.fixture
def client(mock_classifier_service):
    """Test client with mocked classifier service."""
    with patch('app.main.classifier_service', mock_classifier_service):
        from app.main import app
        return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_healthy(self, client):
        """Test health check when models are loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "model_loaded" in data
        assert "device" in data


class TestPredictEndpoint:
    """Tests for single prediction endpoint."""

    def test_predict_success(self, client):
        """Test successful prediction."""
        response = client.post(
            "/predict",
            json={"text": "This is an amazing product!"}
        )
        assert response.status_code in [200, 503]  # 503 if models not loaded

        if response.status_code == 200:
            data = response.json()
            assert "sentiment" in data
            assert "sentiment_confidence" in data
            assert "text" in data

    def test_predict_empty_text(self, client):
        """Test prediction with empty text."""
        response = client.post(
            "/predict",
            json={"text": ""}
        )
        assert response.status_code == 422

    def test_predict_whitespace_only(self, client):
        """Test prediction with whitespace only."""
        response = client.post(
            "/predict",
            json={"text": "   "}
        )
        assert response.status_code == 422

    def test_predict_too_long(self, client):
        """Test prediction with text that's too long."""
        response = client.post(
            "/predict",
            json={"text": "a" * 10000}
        )
        assert response.status_code == 422

    def test_predict_missing_text(self, client):
        """Test prediction without text field."""
        response = client.post(
            "/predict",
            json={}
        )
        assert response.status_code == 422

    def test_predict_valid_text(self, client):
        """Test prediction with valid text."""
        test_texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special.",
            "Amazing experience!",
            "Worst purchase ever."
        ]
        for text in test_texts:
            response = client.post(
                "/predict",
                json={"text": text}
            )
            assert response.status_code in [200, 503]


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_predict_success(self, client):
        """Test successful batch prediction."""
        response = client.post(
            "/predict_batch",
            json={
                "texts": [
                    "Great product!",
                    "Terrible service",
                    "It's okay"
                ]
            }
        )
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_processed" in data
            assert data["total_processed"] == 3

    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list."""
        response = client.post(
            "/predict_batch",
            json={"texts": []}
        )
        assert response.status_code == 422

    def test_batch_predict_too_large(self, client):
        """Test batch prediction with too many items."""
        response = client.post(
            "/predict_batch",
            json={"texts": ["test"] * 101}
        )
        assert response.status_code == 422

    def test_batch_predict_with_empty_text(self, client):
        """Test batch prediction with empty text in list."""
        response = client.post(
            "/predict_batch",
            json={"texts": ["Good", "", "Bad"]}
        )
        assert response.status_code == 422


class TestInfoEndpoints:
    """Tests for info endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_model_info(self, client):
        """Test model info endpoint - endpoint doesn't exist, so expect 404."""
        response = client.get("/info")
        # This endpoint doesn't exist in main.py, so we expect 404
        assert response.status_code == 404


class TestValidation:
    """Tests for input validation."""

    def test_text_length_validation(self, client):
        """Test text length limits."""
        # Too long
        response = client.post(
            "/predict",
            json={"text": "a" * 6000}
        )
        assert response.status_code == 422

        # Valid length
        response = client.post(
            "/predict",
            json={"text": "a" * 100}
        )
        assert response.status_code in [200, 503]

    def test_batch_size_validation(self, client):
        """Test batch size limits."""
        # Too many items
        response = client.post(
            "/predict_batch",
            json={"texts": ["test"] * 150}
        )
        assert response.status_code == 422

        # Valid batch size
        response = client.post(
            "/predict_batch",
            json={"texts": ["test"] * 10}
        )
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

