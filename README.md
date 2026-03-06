# [NOME DO PROJETO]

> Projeto desenvolvido no âmbito da unidade curricular de **[NOME DA UC]**, **[NOME DO CURSO]**, **[NOME DA INSTITUIÇÃO]**.

---

## Descrição do Projeto

Este repositório contém o trabalho de grupo realizado para a UC de **[NOME DA UC]**. O objetivo do projeto é **[DESCREVER O OBJETIVO, ex.: desenvolver e avaliar modelos de machine learning para prever/classificar/regredir ...]**, abordando um problema de **[TIPO DE PROBLEMA, ex.: classificação binária / regressão / clustering / ...]**.

---

## Constituição do Grupo

| Nome | Nº Aluno | Email |
|------|----------|-------|
| [NOME 1] | [Nº ALUNO 1] | [EMAIL 1] |
| [NOME 2] | [Nº ALUNO 2] | [EMAIL 2] |
| [NOME 3] | [Nº ALUNO 3] | [EMAIL 3] |
| [NOME 4] | [Nº ALUNO 4] | [EMAIL 4] |

---

## Organização do Repositório

```
AP_Grupo2/
├── data/
│   ├── raw/                  # Dados brutos, tal como fornecidos originalmente
│   └── processed/            # Dados limpos e transformados, prontos para modelação
├── notebooks/
│   ├── 01_eda.ipynb                        # Análise exploratória de dados (EDA)
│   ├── 02_feature_engineering.ipynb        # Preparação e transformação das variáveis
│   ├── 03_modelagem_treinamento.ipynb      # Treino dos modelos de machine learning
│   └── 04_avaliacao_teste.ipynb            # Avaliação e teste dos modelos
├── src/                      # Código Python reutilizável (módulos e funções auxiliares)
│   ├── data_loading.py       # Carregamento e leitura dos dados
│   ├── preprocessing.py      # Pré-processamento e limpeza dos dados
│   ├── train.py              # Funções de treino dos modelos
│   └── evaluate.py           # Funções de avaliação dos modelos
├── models/                   # Modelos treinados guardados (ex.: ficheiros .pkl, .joblib)
├── reports/
│   └── figures/              # Gráficos e imagens usados no relatório ou apresentação
├── requirements.txt          # Dependências e bibliotecas necessárias para o projeto
└── README.md                 # Documentação principal do repositório
```

### Descrição das Pastas e Notebooks

| Localização | Descrição |
|---|---|
| `data/raw/` | Dados originais, sem qualquer alteração. Devem ser mantidos intactos. |
| `data/processed/` | Dados após limpeza e transformação, usados diretamente nos modelos. |
| `notebooks/01_eda.ipynb` | Análise exploratória: estatísticas descritivas, visualizações e identificação de padrões. |
| `notebooks/02_feature_engineering.ipynb` | Seleção, criação e transformação de variáveis para melhorar os modelos. |
| `notebooks/03_modelagem_treinamento.ipynb` | Definição e treino dos modelos de machine learning. |
| `notebooks/04_avaliacao_teste.ipynb` | Avaliação do desempenho dos modelos com métricas e visualizações. |
| `src/` | Módulos Python com funções reutilizáveis chamadas pelos notebooks. |
| `models/` | Artefactos dos modelos treinados guardados para reutilização. |
| `reports/figures/` | Figuras e gráficos exportados para uso no relatório final. |

---

## Como Correr o Projeto

### 1. Pré-requisitos

- Python 3.9 ou superior
- [Jupyter Notebook](https://jupyter.org/) ou [JupyterLab](https://jupyterlab.readthedocs.io/)

### 2. Instalação das Dependências

Clonar o repositório e instalar as dependências:

```bash
git clone https://github.com/MartimSS/AP_Grupo2.git
cd AP_Grupo2
pip install -r requirements.txt
```

### 3. Ordem de Execução dos Notebooks

Executar os notebooks pela seguinte ordem para reproduzir o pipeline completo:

1. `notebooks/01_eda.ipynb` — Análise exploratória dos dados brutos.
2. `notebooks/02_feature_engineering.ipynb` — Transformação e preparação dos dados.
3. `notebooks/03_modelagem_treinamento.ipynb` — Treino dos modelos.
4. `notebooks/04_avaliacao_teste.ipynb` — Avaliação dos resultados.

> **Nota:** Certifique-se de que os dados brutos estão colocados na pasta `data/raw/` antes de executar o primeiro notebook.

---

## Dependências Principais

As principais bibliotecas utilizadas neste projeto encontram-se listadas no ficheiro `requirements.txt`. Entre elas destacam-se:

- `pandas` — manipulação e análise de dados
- `numpy` — operações numéricas
- `matplotlib` / `seaborn` — visualização de dados
- `scikit-learn` — algoritmos de machine learning
- `jupyter` — ambiente de notebooks interativos

---

*Projeto desenvolvido para fins académicos — [ANO LETIVO]*