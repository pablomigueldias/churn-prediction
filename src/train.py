import sys
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import pipeline_completo


CAMINHO_DADOS = "data/raw/telco_churn.csv"
CAMINHO_MODELO = "models/modelo_churn.pkl"
CAMINHO_COLUNAS = "models/colunas.pkl"
CUSTO_FN = 1500 
CUSTO_FP = 100 
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_DEPTH = 10


def main():
    print("=" * 60)
    print("  PIPELINE DE TREINO — Previsão de Churn")
    print("  Reative Systems")
    print("=" * 60)

    print("\nCarregando e preparando os dados...")
    df = pipeline_completo(CAMINHO_DADOS)
    print(f"   {df.shape[0]} instâncias, {df.shape[1]} colunas.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Treino: {X_treino.shape[0]} | Teste: {X_teste.shape[0]}")

    print("\n Aplicando SMOTE nos dados de treino...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_treino_bal, y_treino_bal = smote.fit_resample(X_treino, y_treino)
    print(f"   Antes: {dict(pd.Series(y_treino).value_counts())}")
    print(f"   Depois: {dict(pd.Series(y_treino_bal).value_counts())}")


    modelos = {
    "Naive Bayes": GaussianNB(),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=MAX_DEPTH),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS) 
    }

    for nome, modelo in modelos.items():
        modelos[nome].fit(X_treino_bal,y_treino_bal)


    print("\n Avaliação no conjunto de teste:")
    y_pred = modelos["Naive Bayes"].predict(X_teste)


    print(f"   Acurácia:  {accuracy_score(y_teste, y_pred):.4f}")
    print(f"   Recall:    {recall_score(y_teste, y_pred):.4f}")
    print(f"   Precisão:  {precision_score(y_teste, y_pred):.4f}")
    print(f"   F1-Score:  {f1_score(y_teste, y_pred):.4f}")

    cm = confusion_matrix(y_teste, y_pred)
    fn = cm[1, 0]
    fp = cm[0, 1]
    custo_total = (fn * CUSTO_FN) + (fp * CUSTO_FP)
    custo_sem_modelo = y_teste.sum() * CUSTO_FN
    economia = custo_sem_modelo - custo_total

    print(f"\n Análise de custo:")
    print(f"   Falsos Negativos (clientes perdidos):    {fn}")
    print(f"   Falsos Positivos (ações desnecessárias):  {fp}")
    print(f"   Custo total dos erros: R$ {custo_total:,.2f}")
    print(f"   Custo SEM modelo:      R$ {custo_sem_modelo:,.2f}")
    print(f"   Economia estimada:     R$ {economia:,.2f} "
          f"({economia/custo_sem_modelo*100:.1f}%)")


    print(f"\n Salvando modelo em {CAMINHO_MODELO}...")
    os.makedirs(os.path.dirname(CAMINHO_MODELO), exist_ok=True)
    joblib.dump(modelo, CAMINHO_MODELO)
    joblib.dump(list(X.columns), CAMINHO_COLUNAS)
    print("Modelo e colunas salvos com sucesso.")

    print("\n" + "=" * 60)
    print("  Pipeline concluído!")
    print("=" * 60)


if __name__ == "__main__":
    main()