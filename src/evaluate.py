"""Módulo para avaliação dos modelos de machine learning."""

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)


def evaluate_classifier(model: BaseEstimator, X_test, y_test) -> dict:
    """Avalia um classificador e devolve as métricas principais.

    Args:
        model: Modelo treinado.
        X_test: Features de teste.
        y_test: Variável alvo de teste.

    Returns:
        Dicionário com accuracy e classification report.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }


def evaluate_regressor(model: BaseEstimator, X_test, y_test) -> dict:
    """Avalia um regressor e devolve as métricas principais.

    Args:
        model: Modelo treinado.
        X_test: Features de teste.
        y_test: Variável alvo de teste.

    Returns:
        Dicionário com MSE e R².
    """
    y_pred = model.predict(X_test)
    return {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
