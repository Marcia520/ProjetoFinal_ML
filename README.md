
# 📘 Projeto Final – Machine Learning I
## Previsão de Inadimplência de Clientes
## Márcia Aparecida Rodrigues de Sousa
# 2. Importação de Bibliotecas
# 1. Introdução

A inadimplência representa um dos principais riscos enfrentados por instituições financeiras, pois impacta diretamente a rentabilidade, a sustentabilidade do negócio e a concessão responsável de crédito. A capacidade de prever, de forma antecipada, clientes com maior probabilidade de não cumprir suas obrigações financeiras permite a adoção de estratégias preventivas, como ajustes de limite, políticas de cobrança e ações de mitigação de risco.

Nesse contexto, técnicas de Machine Learning têm sido amplamente utilizadas para apoiar decisões de crédito, uma vez que permitem identificar padrões complexos em grandes volumes de dados. Este projeto tem como objetivo desenvolver um modelo preditivo de inadimplência, utilizando um dataset público do Kaggle, aplicando um fluxo completo de trabalho em Machine Learning, desde o tratamento dos dados até a avaliação e comparação de modelos.
# Importação de Bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, roc_auc_score
!pip install seaborn
import seaborn as sns

# 3. Definição do Problema de Negócio

O problema de negócio abordado neste projeto consiste em prever se um cliente irá se tornar inadimplente no próximo período de faturamento, com base em informações demográficas, histórico financeiro e comportamento de pagamento.

A variável alvo do problema é binária, indicando se o cliente entrou ou não em inadimplência. A solução proposta pode ser utilizada para apoiar decisões como:

 - concessão ou restrição de crédito;
 - definição de limites de crédito;
 - priorização de ações preventivas de cobrança;
 - apoio à gestão de risco de crédito.
   
Portanto, trata-se de um problema clássico de classificação supervisionada, com alto valor estratégico para o setor financeiro.
# 4. Carregamento do Dataset
# Carregamento do Dataset

df = pd.read_csv(r"C:\Users\Marcia\Documents\ProjetoFinal_ML\credit_card_default.csv")
df.head()
df.shape

# 5. Descrição do Dataset

O dataset utilizado foi obtido na plataforma Kaggle, conhecido como Credit Card Default Dataset. Ele contém aproximadamente 30.000 registros, atendendo ao requisito mínimo de volume de dados para o projeto.

As variáveis incluem:

 - informações demográficas (sexo, escolaridade, estado civil);
 - limite de crédito concedido;
 - histórico de pagamentos anteriores;
 - valores faturados e pagos em períodos anteriores.

A variável alvo indica se o cliente entrou em inadimplência no mês seguinte, permitindo a construção de modelos preditivos com base em dados históricos.
# 6. Pré-processamento

Inicialmente, foi realizada uma análise exploratória dos dados para compreender sua estrutura, tipos de variáveis e distribuição da variável alvo. Observou-se que a classe de inadimplência apresenta desbalanceamento, o que influenciou diretamente a escolha das métricas de avaliação.

O pré-processamento incluiu as seguintes etapas:

- separação entre variáveis explicativas (features) e variável alvo;
- divisão do dataset em conjuntos de treino e teste, utilizando estratificação para preservar a proporção das classes;
- padronização das variáveis numéricas por meio de StandardScaler, especialmente para modelos sensíveis à escala dos dados.
  
Essas transformações garantem maior estabilidade no treinamento dos modelos e evitam vieses decorrentes de escalas diferentes entre as variáveis.
#  Pré-processamento

# Separando variáveis independentes (X) e alvo (y)
X = df.drop("default payment_next_month", axis=1)
y = df["default payment_next_month"]

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Padronização (normalização dos dados)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



df.columns
# 7. Tratamento e Transformação dos Dados
#  Divisão treino/teste com estratificação

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Padronização

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 8. Métricas de Avaliação

Devido ao desbalanceamento das classes e à natureza do problema de risco de crédito, a métrica Recall foi priorizada. Em cenários de inadimplência, é mais crítico identificar corretamente clientes inadimplentes do que classificar erroneamente um cliente adimplente como risco.

Além do Recall, também foram utilizadas:

- F1-Score, para equilibrar precisão e sensibilidade;
- ROC-AUC, para avaliar a capacidade discriminativa dos modelos.

A utilização de múltiplas métricas permite uma avaliação mais robusta e alinhada ao contexto de negócio.

- **Recall**: prioridade para identificar inadimplentes.  
- **F1-Score**: equilíbrio entre precisão e recall.  
- **ROC-AUC**: capacidade discriminativa.
# 9. Modelagem com Machine Learning
Foram utilizados dois estimadores distintos, com o objetivo de comparar modelos de naturezas diferentes:

9.1 Regressão Logística

A Regressão Logística foi utilizada como modelo baseline, por ser amplamente aplicada em problemas de crédito, apresentar boa interpretabilidade e servir como referência inicial de desempenho.

9.2 Random Forest

O segundo modelo adotado foi o Random Forest Classifier, um algoritmo baseado em conjuntos de árvores de decisão. Esse modelo é capaz de capturar relações não lineares e interações complexas entre variáveis, geralmente apresentando melhor desempenho em datasets estruturados.

Ambos os modelos foram treinados e avaliados utilizando o mesmo conjunto de dados, garantindo uma comparação justa.
# Regressão Logística

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)

print("Relatório - Regressão Logística")
print(classification_report(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:,1]))


# Random Forest

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Relatório - Random Forest")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

# 10. Otimização de Hiperparâmetros

Para melhorar o desempenho dos modelos, foi realizada a otimização de hiperparâmetros por meio do método GridSearchCV, utilizando validação cruzada.

Na Regressão Logística, foram ajustados parâmetros como o fator de regularização. No Random Forest, foram testados diferentes valores de número de árvores, profundidade máxima e critérios de divisão.

A otimização resultou em ganhos de desempenho, especialmente no Random Forest, reforçando a importância dessa etapa no fluxo de Machine Learning.
#### GridSearch para Regressão Logística
param_grid_log = {"C": [0.01, 0.1, 1, 10]}
grid_log = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                        param_grid_log, cv=5, scoring="recall", n_jobs=-1)
grid_log.fit(X_train_scaled, y_train)
print("Melhores parâmetros Logística:", grid_log.best_params_)

#### GridSearch para Random Forest (simplificado para evitar travar)
param_grid_rf = {
    "n_estimators": [100],
    "max_depth": [10, None],
    "min_samples_split": [2]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42),
                       param_grid_rf, cv=3, scoring="recall", n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("Melhores parâmetros RF:", grid_rf.best_params_)

#### Modelo final de Regressão Logística com melhor parâmetro
best_C = grid_log.best_params_["C"]

log_reg = LogisticRegression(C=best_C, random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Avaliação
y_pred = log_reg.predict(X_test_scaled)

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Importar métricas necessárias
from sklearn.metrics import recall_score, f1_score, roc_auc_score
import pandas as pd

# Predições dos melhores modelos
y_pred_best_log = best_log.predict(X_test_scaled)
y_pred_best_rf = best_rf.predict(X_test)

# Cálculo das métricas principais
results = {
    "Modelo": ["Regressão Logística", "Random Forest"],
    "Recall": [
        recall_score(y_test, y_pred_best_log),
        recall_score(y_test, y_pred_best_rf)
    ],
    "F1-Score": [
        f1_score(y_test, y_pred_best_log),
        f1_score(y_test, y_pred_best_rf)
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, best_log.predict_proba(X_test_scaled)[:,1]),
        roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1])
    ]
}

# Criar DataFrame para visualização
df_results = pd.DataFrame(results)
df_results



# 11. Comparação dos Modelos

Após a avaliação dos modelos e a otimização dos hiperparâmetros, os resultados indicaram que o Random Forest apresentou melhor desempenho, especialmente em termos de Recall e ROC-AUC.

Dessa forma, o Random Forest foi escolhido como o modelo final, por demonstrar maior capacidade de identificar clientes inadimplentes, alinhando-se ao objetivo principal do problema de negócio.
# 12. Avaliação Crítica da Solução

Este projeto demonstrou a aplicação de um fluxo completo de Machine Learning para previsão de inadimplência, usando Random Forest. Ele resolveu adequadamente o problema propossto, pois apresentou melhor Recall e F1-Score, identificando inadimplentes com maior precisão.

Embora o modelo apresente bons resultados, sua implantação em produção é necessário a integração a um pipeline de dados,  monitoramento contínuo de desempenho; análise de viés e equidade; tratamento de data drift; integração com sistemas corporativos e governança de modelos.
A Regressão Logística pode ser usada como baseline interpretável, enquanto o Random Forest serve como modelo principal. 

Portanto, o modelo é tecnicamente viável, mas requer cuidados adicionais para uso em ambiente produtivo.
# 13. Conclusão

Este projeto demonstrou a aplicação de um fluxo completo de Machine Learning para previsão de inadimplência, desde o tratamento dos dados até a seleção do modelo final. Os resultados evidenciam o potencial do uso de técnicas de aprendizado de máquina para apoiar decisões de crédito e gestão de risco em instituições financeiras.

Como trabalhos futuros, sugere-se a inclusão de novos atributos comportamentais, técnicas de balanceamento de classes e a avaliação de modelos adicionais, como Gradient Boosting, visando aprimorar ainda mais o desempenho da solução.

