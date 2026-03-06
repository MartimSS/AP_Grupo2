"""Módulo para carregamento e leitura dos dados."""

import pandas as pd


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Carrega dados brutos a partir de um ficheiro CSV.

    Args:
        filepath: Caminho para o ficheiro de dados.

    Returns:
        DataFrame com os dados carregados.
    """
    return pd.read_csv(filepath)
