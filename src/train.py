"""Módulo para treino dos modelos de machine learning."""

import joblib
from sklearn.base import BaseEstimator


def train_model(model: BaseEstimator, X_train, y_train) -> BaseEstimator:
    """Treina um modelo com os dados fornecidos.

    Args:
        model: Instância de um estimador scikit-learn.
        X_train: Features de treino.
        y_train: Variável alvo de treino.

    Returns:
        Modelo treinado.
    """
    model.fit(X_train, y_train)
    return model


def save_model(model: BaseEstimator, filepath: str) -> None:
    """Guarda o modelo treinado em disco.

    Args:
        model: Modelo treinado.
        filepath: Caminho de destino (ex.: 'models/model.pkl').
    """
    joblib.dump(model, filepath)
