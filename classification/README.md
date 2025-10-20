# 🏦 Análise e Predição de Aprovação de Empréstimos

Projeto de **Machine Learning aplicado à concessão de crédito**, com foco em identificar os fatores que influenciam a aprovação de empréstimos e comparar diferentes modelos de classificação.

---

## 📊 1. Visão Geral do Projeto

O objetivo é **prever a aprovação (`Loan_Approved`)** com base em informações financeiras e demográficas dos solicitantes, como renda, score de crédito, valor do empréstimo e endividamento.

**Etapas principais:**
1. Análise estatística descritiva e exploração dos dados  
2. Pré-processamento e codificação de variáveis  
3. Treinamento e avaliação de modelos supervisionados  
4. Interpretação e comparação de desempenho  
5. Análise de importância das variáveis e curva de aprendizado  

---

## 📈 2. Resumo Estatístico

| Variável | Interpretação Estatística | Insight Analítico |
|-----------|---------------------------|-------------------|
| **Applicant_Income** | Média ≈ 10.8k, desvio ≈ 4.9k | Renda do titular com alta dispersão e forte assimetria positiva. |
| **Coapplicant_Income** | Média ≈ 5.0k | Co-solicitante (geralmente cônjuge) complementa a renda familiar. |
| **Credit_Score** | Média ≈ 676 | Score médio-alto, fator chave na aprovação. |
| **DTI_Ratio** | Média ≈ 0.35 (35%) | Endividamento moderado; valores > 0.5 indicam maior risco. |
| **Loan_Amount** | Média ≈ 20.5k | Aprovados tendem a solicitar valores próximos à mediana (~R$18–20k). |
| **Collateral_Value** | Média ≈ 24.7k | Valor variável; não é determinante isolado na aprovação. |

---

## ⚖️ 3. Distribuições e Perfil da Base

- **Loan_Approved** → 26% aprovados, 74% rejeitados ➜ base **desbalanceada**.  
- **Gênero** → 62% homens, 38% mulheres.  
- **Estado civil** → 64% casados.  
- **Idade média** → 40 anos (intervalo 21–59).  

💡 *Público predominantemente adulto, casado e de renda média, refletindo perfil de estabilidade financeira.*

---

## 🔍 4. Correlações com Aprovação

| Variável | Correlação | Interpretação |
|-----------|-------------|---------------|
| **Credit_Score** | +0.45 | Quanto maior o score, maior a chance de aprovação. |
| **DTI_Ratio** | -0.44 | Endividamento alto reduz probabilidade de aprovação. |
| **Applicant_Income** | +0.12 | Renda ajuda, mas não é fator decisivo isoladamente. |
| **Loan_Amount** | -0.13 | Valores altos tendem a ser mais rejeitados. |

✅ Nenhuma variável apresenta multicolinearidade (VIF < 2).

---

## ⚙️ 5. Pré-processamento

- **LabelEncoder** → variáveis binárias/ordinais (ex.: `Gender`, `Marital_Status`).  
- **One-Hot Encoding** → variáveis nominais (ex.: `Loan_Purpose`, `Employer_Category`).  
- **Padronização (StandardScaler)** → centraliza e escala variáveis numéricas.  
- **Divisão dos dados:** 60% treino / 40% teste (`random_state=42`).  

---

## 🤖 6. Modelagem Preditiva

Modelos treinados:
- **Regressão Logística**
- **SVM (kernel RBF)**
- **Random Forest (200 árvores)**

Cada modelo foi avaliado em termos de acurácia, precisão, recall, F1-score, AUC e calibração.

---

## 📊 7. Desempenho dos Modelos

| Modelo | Acurácia | Precisão | Recall | F1 | Observações |
|--------|-----------|-----------|--------|----|-------------|
| **Regressão Logística** | 0.84 | 0.81 | 0.66 | 0.73 | Boa explicabilidade, leve perda de recall. |
| **SVM (RBF)** | 0.83 | 0.85 | 0.58 | 0.69 | Conservador, evita falsos positivos. |
| **Random Forest** | 🥇 0.90 | 0.88 | 0.80 | 0.84 | Melhor equilíbrio geral, robusto e não linear. |

---

## 📈 8. Avaliação Detalhada

| Métrica | Regressão Logística | SVM | Random Forest |
|----------|----------------------|-----|----------------|
| **AUC (ROC)** | 0.91 | 0.93 | 🥇 0.96 |
| **F1-score** | 0.78 | 0.80 | 🥇 0.91 |
| **Precisão (1)** | 0.76 | 0.73 | 🥇 0.85 |
| **Recall (1)** | 0.80 | 0.89 | 🥇 0.98 |
| **Brier Score** | 0.115 | 0.105 | 🥇 0.082 |

🔹 **Random Forest** → melhor performance global  
🔹 **Regressão Logística** → melhor calibração  
🔹 **SVM** → bom recall, mas menos interpretável

---

## 🌡️ 9. Calibração e Curva ROC

- **Regressão Logística**: calibração quase perfeita, AUC = 0.91  
- **SVM**: AUC = 0.93, AP = 0.85, leve subestimação em extremos  
- **Random Forest**: AUC = 0.96, bem calibrado após threshold ótimo (0.32)

---

## 🌳 10. Importância das Variáveis (Random Forest)

| Variável mais relevante | Impacto |
|--------------------------|---------|
| **Credit_Score** | Mais importante na decisão de aprovação |
| **DTI_Ratio** | Endividamento elevado reduz probabilidade |
| **Applicant_Income** | Renda ajuda, mas depende do score |
| **Loan_Amount** | Empréstimos muito altos tendem à rejeição |

📊 O modelo combina múltiplas dimensões de risco, sem depender apenas da renda.

---

## 📚 11. Curva de Aprendizado

- AUC de **validação estabiliza em 0.97–0.98** após ~200 observações.  
- Gap mínimo entre treino e validação → **excelente generalização**.  
- Pouco risco de overfitting.

---

## 🧩 12. Conclusão Geral

| Objetivo | Modelo Ideal | Motivo |
|-----------|---------------|--------|
| **Alta performance preditiva** | 🥇 **Random Forest** | Melhor AUC, F1 e Recall |
| **Explicabilidade e análise de risco** | **Regressão Logística** | Calibração e interpretabilidade |
| **Triagem ampla (sensível)** | **SVM** | Recall elevado, cauteloso na aprovação |

🔹 **Melhor modelo final:** `Random Forest`  
🔹 **AUC = 0.96**, **F1 = 0.91**, **generalização excelente**  

---

## 🧠 13. Próximos Passos

- Aplicar **SMOTE** ou `class_weight='balanced'` para lidar com o desbalanceamento.  
- Testar **XGBoost ou LightGBM** para comparação.  
- Implementar **explicabilidade local (SHAP values)**.  
- Deploy em API (Flask/FastAPI) para uso real.

---

## 🧾 Autor
**Gepeto**  
📍 UFSM — Departamento de Estatística  
📧 Contato acadêmico / profissional  
🔗 [LinkedIn / GitHub / Lattes – opcional]

---

## 🧠 Tecnologias Principais
- Python 3.13  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn, statsmodels  
- Jupyter Notebook

---

> _“Modelos bem calibrados e interpretáveis são tão importantes quanto modelos precisos.”_
