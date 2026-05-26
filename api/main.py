import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

CAMINHO_MODELO = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_churn.pkl")
CAMINHO_COLUNAS = os.path.join(os.path.dirname(__file__), "..", "models", "colunas.pkl")

try:
    modelo = joblib.load(CAMINHO_MODELO)
    colunas_esperadas = joblib.load(CAMINHO_COLUNAS)
    modelo_carregado = True
except FileNotFoundError:
    modelo = None
    colunas_esperadas = None
    modelo_carregado = False

class DadosCliente(BaseModel):

    gender: str = Field(
        ..., example="Female",
        description="Gênero do cliente: Male ou Female"
    )
    SeniorCitizen: int = Field(
        ..., example=0,
        description="É idoso? 0 = Não, 1 = Sim"
    )
    Partner: str = Field(
        ..., example="Yes",
        description="Tem cônjuge? Yes ou No"
    )
    Dependents: str = Field(
        ..., example="No",
        description="Tem dependentes? Yes ou No"
    )
    tenure: int = Field(
        ..., example=12,
        description="Meses como cliente"
    )
    PhoneService: str = Field(
        ..., example="Yes",
        description="Tem serviço de telefone? Yes ou No"
    )
    MultipleLines: str = Field(
        ..., example="No",
        description="Tem múltiplas linhas? Yes, No ou No phone service"
    )
    InternetService: str = Field(
        ..., example="Fiber optic",
        description="Tipo de internet: DSL, Fiber optic ou No"
    )
    OnlineSecurity: str = Field(
        ..., example="No",
        description="Tem segurança online? Yes, No ou No internet service"
    )
    OnlineBackup: str = Field(
        ..., example="No",
        description="Tem backup online? Yes, No ou No internet service"
    )
    DeviceProtection: str = Field(
        ..., example="No",
        description="Tem proteção de dispositivo? Yes, No ou No internet service"
    )
    TechSupport: str = Field(
        ..., example="No",
        description="Tem suporte técnico? Yes, No ou No internet service"
    )
    StreamingTV: str = Field(
        ..., example="No",
        description="Tem streaming TV? Yes, No ou No internet service"
    )
    StreamingMovies: str = Field(
        ..., example="No",
        description="Tem streaming de filmes? Yes, No ou No internet service"
    )
    Contract: str = Field(
        ..., example="Month-to-month",
        description="Tipo de contrato: Month-to-month, One year ou Two year"
    )
    PaperlessBilling: str = Field(
        ..., example="Yes",
        description="Fatura digital? Yes ou No"
    )
    PaymentMethod: str = Field(
        ..., example="Electronic check",
        description="Método de pagamento"
    )
    MonthlyCharges: float = Field(
        ..., example=70.35,
        description="Valor mensal cobrado"
    )
    TotalCharges: float = Field(
        ..., example=840.50,
        description="Total já cobrado"
    )

app = FastAPI(
    title="API de Previsão de Churn",
    description=(
        "API que recebe dados de um cliente e retorna a probabilidade "
        "de churn (evasão). Desenvolvida pela Reative Systems."
    ),
    version="1.0.0"
)


@app.get("/")
def raiz():
    return {
        "mensagem": "API de Previsão de Churn — Reative Systems",
        "status": "online",
        "modelo_carregado": modelo_carregado,
        "docs": "Acesse /docs para a documentação interativa."
    }


@app.get("/health")
def health():
    if not modelo_carregado:
        raise HTTPException(
            status_code=503,
            detail="Modelo não encontrado. Rode 'python src/train.py' primeiro."
        )
    return {"status": "healthy"}


@app.post("/prever")
def prever_churn(cliente: DadosCliente):
    if not modelo_carregado:
        raise HTTPException(
            status_code=503,
            detail="Modelo não encontrado. Rode 'python src/train.py' primeiro."
        )

    dados_dict = cliente.model_dump()
    df_cliente = pd.DataFrame([dados_dict])

    df_cliente = pd.get_dummies(df_cliente, drop_first=True)
    df_cliente = df_cliente.astype(
        {col: "int" for col in df_cliente.select_dtypes("bool").columns}
    )

    for col in colunas_esperadas:
        if col not in df_cliente.columns:
            df_cliente[col] = 0

    df_cliente = df_cliente[colunas_esperadas]

    previsao = modelo.predict(df_cliente)[0]
    probabilidade = modelo.predict_proba(df_cliente)[0][1]


    if probabilidade >= 0.7:
        risco = "Alto"
    elif probabilidade >= 0.4:
        risco = "Médio"
    else:
        risco = "Baixo"

    return {
        "previsao": "Churn" if previsao == 1 else "Não Churn",
        "probabilidade_churn": round(float(probabilidade), 4),
        "risco": risco
    }
