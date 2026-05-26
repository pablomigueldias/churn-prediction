# Previsão de Churn de Clientes

> Projeto de Machine Learning para identificar clientes com risco de abandonar a empresa,
> permitindo ações de retenção antes da perda.

---

## Contexto e Motivação

Este projeto foi desenvolvido no contexto da **Reative Systems**, empresa de sistemas web,
automação e IA. Demonstra um pipeline completo de Machine Learning aplicado a um problema
real de negócio: a **previsão de churn** (evasão de clientes).

> **Nota sobre os dados:** o projeto utiliza um dataset público de telecomunicações
> (*Telco Customer Churn*) como proxy. O pipeline foi desenhado para ser reaplicável
> a dados reais de clientes assim que houver volume suficiente.

**Por que churn importa?** Conquistar um cliente novo custa muito mais do que reter um
existente. Um modelo que antecipa quem vai sair permite agir a tempo — e isso é dinheiro.

---

## Problema de Negócio

A Reactive Systems trabalha com receita recorrente (assinaturas, contratos mensais, SaaS) e enfrenta um desafio constante: a evasão de clientes. Identificar com antecedência quais clientes têm maior probabilidade de cancelar permite agir antes da perda — com ofertas de retenção, atendimento proativo ou ajustes no serviço.

Sem um modelo preditivo, a empresa só descobre que perdeu o cliente **depois que ele já saiu**. Com o modelo, é possível agir **enquanto ainda dá tempo**.

### Custo dos erros

Nem todo erro do modelo custa o mesmo:

- **Falso Negativo (FN):** o modelo diz "vai ficar", mas o cliente sai. A empresa perde a receita do cliente e precisa investir para conquistar um substituto. **Custo estimado: R$ 1.500 por cliente.**

- **Falso Positivo (FP):** o modelo diz "vai sair", mas o cliente ficaria. A empresa gasta com uma ação de retenção desnecessária (descontos, ligações). **Custo estimado: R$ 100 por cliente.**

Como perder um cliente custa **15 vezes mais** do que uma ação de retenção à toa, o modelo deve priorizar **não deixar passar** quem vai sair — ou seja, o **recall** é mais importante que a precisão neste projeto.

---

## Metodologia

O projeto segue um pipeline estruturado:

1. **Análise Exploratória (EDA)** — entender os dados, distribuições e o desbalanceamento.
2. **Engenharia de Atributos** — tratar dados faltantes, tipos incorretos e codificar categorias.
3. **Balanceamento** — aplicação de SMOTE para lidar com o desbalanceamento da classe.
4. **Modelagem** — treino e comparação de 3 algoritmos (Naive Bayes, Árvore de Decisão, Random Forest).
5. **Avaliação por Custo** — análise dos erros traduzida em impacto financeiro.
6. **API** — modelo servido via FastAPI para consumo em produção.

---

## Análise Exploratória (EDA)

### Atributos que NÃO influenciam o churn

- **Gênero** — distribuição praticamente igual entre os que ficam e os que saem. Não é fator relevante.

### Atributos de perfil do cliente

- Clientes **idosos** têm proporção de churn maior que jovens, apesar de serem minoria no dataset.
- Clientes **sem cônjuge** e **sem dependentes** saem mais — possivelmente porque têm menos vínculo que dificulte a troca de fornecedor.

### Atributos de contrato

- **Contrato mensal** tem churn muito maior que contratos de 1 ou 2 anos. Sem compromisso de longo prazo, a barreira de saída é baixa.
- **Pagamento por Electronic Check** tem churn mais alto (~45%). Métodos automáticos (cartão, transferência) retêm mais.
- **Paperless Billing** (fatura digital) está associado a mais churn (~50%). Pode indicar que esses clientes são mais "digitais" e comparam mais ofertas online.

### Atributos de serviço

- Clientes com **fibra óptica** saem mais que os com DSL — possivelmente pelo preço mais alto.
- Clientes **sem** suporte técnico, segurança online, backup online e proteção de dispositivo saem mais. Esses serviços funcionam como "âncoras" — quanto mais serviços o cliente tem, maior o custo de saída.
- Streaming (TV e Movies) tem pouca influência isolada.

### Atributos financeiros

- Mensalidades na faixa de **R$ 80-100** concentram mais churn.
- Clientes com **pouco tempo de casa** (tenure baixo) saem mais — os primeiros meses são os mais críticos.

### Perfil de risco identificado

> O cliente típico que dá churn tem: **contrato mensal**, **pouco tempo de casa**, **paga caro**, **poucos serviços adicionais**, **paga por boleto eletrônico** e **não tem cônjuge/dependentes**. É alguém com pouco vínculo com a empresa — tanto contratual quanto de serviços.

---

## Resultados

| Modelo | Acurácia | Recall | Precisão | F1-Score |
|---|---|---|---|---|
| Naive Bayes | 0.6507 | 0.8717 |0.4234| 0.5699 |
| Árvore de Decisão | 0.7880 | 0.5169 |0.6210| 0.5642 |
| Random Forest | 0.7889 | 0.4973 | 0.6298 | 0.5558 |

> Sem SMOTE, o recall ficou baixo em todos os modelos — o dataset desbalanceado faz o modelo "se acomodar" prevendo a classe majoritária.

### Comparação dos modelos (com SMOTE)

| Modelo | Acurácia | Recall | Precisão | F1-Score |
|---|---|---|---|---|
| Naive Bayes | 0.7104 | 0.8021 | 0.4732 | 0.5952 |
| Árvore de Decisão | 0.7421 | 0.6667 | 0.5109 | 0.5785 |
| Random Forest | 0.7757 | 0.6096 | 0.5729 | 0.5907 |

> Com o SMOTE, o recall subiu significativamente — o modelo passou a identificar mais clientes em risco, mesmo ao custo de uma leve queda na acurácia.

### Análise de custo

| Cenário | Custo |
|---|---|
| **Sem modelo** (todos os churns passam despercebidos) | R$ 841,500.00 |
| **Com o melhor modelo** | R$ 216,600.00 |
| **Economia estimada** | R$ 624,900.00 (74.3% de redução) |

---

## Conclusão

O modelo **Naive Bayes** com SMOTE foi o que apresentou o **menor custo total de erros**, identificando a maior parte dos clientes em risco de churn.

Aplicado à base de teste, o modelo representaria uma **economia estimada de R$ 624,900.00** em comparação ao cenário sem nenhum modelo — uma **redução de 74.3%** nas perdas com evasão de clientes.

### Limitações

- O modelo foi treinado com dados de telecomunicações, não com dados reais da Reative Systems. Os padrões de churn podem variar entre setores.
- Os custos de FN (R$ 1.500) e FP (R$ 100) são estimativas — em produção, seriam calibrados com dados financeiros reais.
- O modelo não considera fatores externos (concorrência, sazonalidade, mudanças de mercado).

### Melhorias futuras

- Aplicar **tuning de hiperparâmetros** (Grid Search / Random Search) para otimizar o modelo.
- Testar algoritmos adicionais (Gradient Boosting, XGBoost).
- Implementar **monitoramento do modelo** em produção para detectar degradação de performance.
- Re-treinar com dados reais da Reative Systems quando houver volume suficiente.

---
## Estrutura do Projeto

```
churn-prediction/
├── data/
│   ├── raw/               # dados originais, intocados
│   └── processed/         # dados após limpeza
├── notebooks/
│   ├── eda.ipynb       # análise exploratória
│   └── modelagem.ipynb # treino e avaliação dos modelos
├── src/
│   ├── preprocessing.py   # funções de limpeza e preparação
│   └── train.py           # script que treina e salva o modelo
├── models/                # modelo treinado (.pkl)
├── api/
│   └── main.py            # API FastAPI para servir o modelo
├── README.md
└── requirements.txt
```

---

## Como Executar

### 1. Configurar o ambiente

```bash
git clone https://github.com/pablomigueldias/churn-prediction.git
cd churn-prediction

python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Treinar o modelo

```bash
python src/train.py
```

### 3. Subir a API

```bash
uvicorn api.main:app --reload
```

Acesse **http://127.0.0.1:8000/docs** para a documentação interativa.

### Exemplo de uso da API

```bash
curl -X POST http://127.0.0.1:8000/prever \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 190.0
  }'
```

---

## Tecnologias

- **Python** — linguagem principal
- **pandas / NumPy** — manipulação de dados
- **scikit-learn** — modelos de Machine Learning
- **imbalanced-learn** — tratamento de desbalanceamento (SMOTE)
- **Matplotlib / Seaborn** — visualização
- **FastAPI** — API REST para servir o modelo
- **joblib** — serialização do modelo treinado

---

## Autor


**Seu Nome**
- GitHub: [pablomigueldias](https://github.com/pablomigueldias)
- LinkedIn: [Pablo Ortiz](https://www.linkedin.com/in/pablo-miguel-dias-ortiz/)