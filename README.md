## 📘 Projeto Final – Machine Learning I
### Previsão de Inadimplência de Clientes
#### Autora: Márcia Aparecida Rodrigues de Sousa

### Google Colab

Clique no botão abaixo para abrir e executar o notebook diretamente no Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Marcia520/ProjetoFinal_ML/blob/main/ProjetoFinal%20(3).ipynb)

---

### 📌 Introdução
Este projeto tem como objetivo prever a inadimplência de clientes de cartão de crédito utilizando técnicas de Machine Learning.  
A solução foi desenvolvida com base em um dataset público do Kaggle, aplicando um fluxo completo de ML: desde o tratamento dos dados até a comparação de modelos e escolha do melhor.

---

### 📊 Dataset
- Fonte: [Credit Card Default Dataset – Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)  
- Registros: ~30.000 clientes  
- Variável alvo: **default payment_next_month** (inadimplência no próximo mês)

---

#### ⚙️ Fluxo do Projeto
1. Importação e análise exploratória dos dados  
2. Pré-processamento (estratificação, padronização)  
3. Modelagem com dois estimadores:
   - Regressão Logística (baseline)
   - Random Forest (modelo mais robusto)
4. Otimização de hiperparâmetros com GridSearchCV  
5. Avaliação com métricas Recall, F1-Score e ROC-AUC  
6. Comparação dos modelos e escolha do melhor  

---

#### 📈 Resultados Comparativos

| Modelo               | Recall | F1-Score | ROC-AUC |
|----------------------|--------|----------|---------|
| Regressão Logística  | 0.62   | 0.58     | 0.74    |
| Random Forest        | 0.71   | 0.65     | 0.82    |

---

### ▶️ Como Executar Localmente

1. Clone este repositório:
   ```bash
   git clone https://github.com/Marcia520/ProjetoFinal_ML.git
   cd ProjetoFinal_ML
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Abra o notebook:
   ```bash
   jupyter notebook "ProjetoFinal (1).ipynb"
   ```

4. Execute célula por célula para reproduzir os resultados.

---

### 🚀 Conclusão
O projeto demonstrou que o **Random Forest** é o modelo mais eficaz para prever inadimplência, com melhor Recall e ROC-AUC.  
Para produção, recomenda-se integração com pipeline de dados, monitoramento contínuo e análise de viés.

```
