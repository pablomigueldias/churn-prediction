# Previsão de Churn de Clientes

> Projeto de Machine Learning para identificar clientes com risco de abandonar a empresa,
> permitindo ações de retenção antes da perda.


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

A Rative Systems trabalha com uma receita recorrente como assinaturas, contratos mensais, Saas que enfrenta um desafio constante que seria a evasão de cliente. poder identificar com antecedência quais clientes têm a maior probabilidade de cancelar permite agir antes da perda, com ofertas de retenção, atendimentos ou ajuste no serviço, sem modelo a empresa só descobre que perdeu o cliente depois que ele ja saiu, com este modelo é possivel agir enquanto dá tempo.

**Custo dos erros**
Nem todo erro custa o mesmo:

- Falso Negativo: O modelo diz **"vai ficar"**, mas o cliente sai. A empresa perde a receita do cliente e precisa investir para conquistar um substito. que o custo estimado seria de R$1500,00 por cliente

- Falso Positivo: O modelo diz **"vai sair"**, mas o cliente ficaria: a empresa gastaria uma acão de retenção desnecessária com descontos e ligações. um custo estimado de R$ 100 por cliente.

como perder o cliente custa muito mais que uma retenção á toa, o modelo deve priorizar não deixar passar quem vai sair -- ou seja, o recall é mais importante que a precisão.

---

## Analise

### EDA

Atributos que NÃO influenciam o churn:

- Gênero — distribuição praticamente igual entre os que ficam e os que saem. Não é fator relevante.

Atributos de perfil do cliente:

- Clientes idosos têm proporção de churn maior que jovens, apesar de serem minoria no dataset.
- Clientes sem cônjuge e sem dependentes saem mais — possivelmente porque têm menos "vínculo" que dificulte a troca de fornecedor.

Atributos de contrato (como o cliente compra):

- Contrato mensal tem churn muito maior que contratos de 1 ou 2 anos. Faz sentido: sem compromisso de longo prazo, a barreira de saída é baixa.
- Pagamento por Electronic Check tem churn mais alto (~45%). Métodos automáticos (cartão, transferência) retêm mais.
- Paperless Billing (fatura digital) está associado a mais churn (~50%). Pode indicar que esses clientes são mais "digitais" e comparam mais ofertas online.

Atributos de serviço (o que o cliente usa):

- Clientes com fibra óptica saem mais que os com DSL. Pode ser pelo preço mais alto.
- Clientes sem suporte técnico, segurança online, backup online e proteção de dispositivo saem mais. Esses serviços funcionam como "âncoras" — quanto mais serviços o cliente tem, mais "custo de saída" ele tem ao cancelar.
- Streaming (TV e Movies) tem pouca influência isolada.

Atributos financeiros (quanto custa):

Mensalidades na faixa de R$ 80-100 concentram mais churn.
Clientes com pouco tempo de casa (tenure baixo) saem mais — os primeiros meses são os mais críticos.


### Conclusão

O Cliente que da churn é: contrato mensal que tem pouco tempo de casa, paga caro que tem pouco serviços adicionais, paga por boleto eletronico e nao tem conjuge/dependentes. Um cliente com pouco vinculos com a empresa tanto em serviço e contratual.

---

## Estrutura do Projeto

```
churn-prediction/
├── data/
│   ├── raw/              # dados originais, intocados
│   └── processed/        # dados após limpeza
├── notebooks/
│   ├── 01_eda.ipynb      # análise exploratória
│   └── 02_modelagem.ipynb# treino e avaliação dos modelos
├── src/
│   ├── preprocessing.py  # funções de limpeza e preparação
│   └── train.py          # script que treina e salva o modelo
├── models/               # modelo treinado (.pkl)
├── api/
│   └── main.py           # API FastAPI para servir o modelo
├── README.md
└── requirements.txt             # guia de configuração do ambiente
```

---
