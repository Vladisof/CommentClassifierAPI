class ClassifierException(Exception):
    """Base exception for classifier errors."""
    pass


class ModelNotFoundError(ClassifierException):
    """Raised when a model file is not found."""
    pass


class ModelLoadError(ClassifierException):
    """Raised when model loading fails."""
    pass


class PredictionError(ClassifierException):
    """Raised when prediction fails."""
    pass


class ValidationError(ClassifierException):
    """Raised when input validation fails."""
    pass


class DataError(ClassifierException):
    """Raised when data processing fails."""
    pass


class TrainingError(ClassifierException):
    """Raised when model training fails."""
    pass


class ConfigurationError(ClassifierException):
    """Raised when configuration is invalid."""
    pass

