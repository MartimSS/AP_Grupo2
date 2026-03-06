"""Módulo para pré-processamento e limpeza dos dados."""

import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas duplicadas do DataFrame."""
    return df.drop_duplicates()


def fill_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Preenche valores em falta usando a estratégia indicada.

    Args:
        df: DataFrame de entrada.
        strategy: Estratégia de imputação ('mean', 'median' ou 'mode').

    Returns:
        DataFrame com valores em falta preenchidos.

    Raises:
        ValueError: Se a estratégia indicada não for suportada.
    """
    supported = {"mean", "median", "mode"}
    if strategy not in supported:
        raise ValueError(
            f"Estratégia '{strategy}' não suportada. Escolha entre: {supported}."
        )
    numeric_cols = df.select_dtypes(include="number").columns
    if strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "mode":
        mode_values = df[numeric_cols].mode()
        if not mode_values.empty:
            df[numeric_cols] = df[numeric_cols].fillna(mode_values.iloc[0])
    return df
