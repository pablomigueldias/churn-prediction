import pandas as pd

def carregar_dados(caminho:str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    return df

def limpar_dados(df:pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        mediana = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(mediana)

    return df

def preparar_para_modelo(df:pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "Churn" in df.columns:
        df['Churn'] = df['Churn'].map({'Yes':1, 'No': 0})

    df = pd.get_dummies(df, drop_first=True)

    df = df.astype({col: "int" for col in df.select_dtypes("bool").columns})

    return df

def pipeline_completo(caminho:str) -> pd.DataFrame:
    df = carregar_dados(caminho)
    df = limpar_dados(df)
    df = preparar_para_modelo(df)

    return df

if __name__ == "__main__":
    dados = pipeline_completo("data/raw/telco_churn.csv")
    print("Pré-processamento concluído")
    print(f"Formato final: linhas {dados.shape[0]} e colunas {dados.shape[1]}")
    print(f"Colunas: {list(dados.columns)}")

