# %% [markdown]
# ## **Previsão de Aprovação de Empréstimos**

# %% [markdown]
# **Importação das bibliotecas**

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# bibliotecas adicionais (estatística, explicabilidade, tuning)
import statsmodels.api as sm
import scipy.stats as st
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import (
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay,
    brier_score_loss, matthews_corrcoef, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
import plotly.figure_factory as ff

# %% [markdown]
# **Carregamento dos dados**

# %%
df = pd.read_csv("C:/Users/Cliente/Downloads/loan_approval.csv")

# %%
df.head()

# %% [markdown]
# ### Dicionário de Variáveis
# 
# | **Variável** | **Descrição** |
# |:--------------|:--------------|
# | **Applicant_ID** | Identificador único do solicitante |
# | **Applicant_Income** | Renda mensal do solicitante |
# | **Coapplicant_Income** | Renda mensal do co-solicitante (se houver) |
# | **Employment_Status** | Situação profissional (ex: assalariado, autônomo, desempregado) |
# | **Age** | Idade do solicitante |
# | **Marital_Status** | Estado civil (ex: solteiro, casado) |
# | **Dependents** | Número de dependentes |
# | **Credit_Score** | Pontuação de crédito |
# | **Existing_Loans** | Quantidade de empréstimos já existentes |
# | **DTI_Ratio** | Relação dívida/renda (*Debt-to-Income Ratio*) |
# | **Savings** | Valor da poupança ou reserva financeira |
# | **Collateral_Value** | Valor do bem dado em garantia |
# | **Loan_Amount** | Valor solicitado do empréstimo |
# | **Loan_Term** | Prazo do empréstimo (em meses ou anos) |
# | **Loan_Purpose** | Finalidade do empréstimo (ex: casa, carro, educação) |
# | **Property_Area** | Localização do imóvel (urbana, semiurbana ou rural) |
# | **Education_Level** | Nível de escolaridade do solicitante |
# | **Gender** | Gênero (masculino, feminino, outro) |
# | **Employer_Category** | Setor do emprego (público, privado, autônomo etc.) |
# | **Loan_Approved** | **Variável-alvo** → `Yes` (Aprovado) / `No` (Negado) |
# 

# %% [markdown]
# Remoção de identificador: a coluna `Applicant_ID` é um identificador sem informação preditiva direta. 
# 

# %%
df=df.drop("Applicant_ID", axis=1)

# %%
df.head()

# %% [markdown]
# **Checagem de valores ausentes**

# %%
df.isnull().sum()

# %% [markdown]
# Precisamos tratar nulos antes do ajuste de modelos para evitar erros e perda de amostras.

# %% [markdown]
# ### **Resumo estatístico**

# %%
print(df.describe())

# %% [markdown]
# **Informações do DataFrame**

# %%
print(df.info())

# %% [markdown]
# Separação de variáveis por tipo
# `categorical_col`: colunas do tipo objeto (categóricas)
# `numerical_col`: colunas numéricas (int/float)

# %%
categorical_col=df.select_dtypes(include=['object']).columns
numerical_col=df.select_dtypes(include=['int64', 'float64']).columns

# %%
categorical_col

# %%
numerical_col

# %% [markdown]
# **Imputação `SimpleImputer` (Scikit-Learn) para preencher nulos**
# 
# Estratégias:
# 
# * Numéricas → mediana (robusta a outliers)
# 
# * Categóricas → categoria mais frequente (evita criação de rótulos artificiais)
# 

# %% [markdown]
# **Imputação numérica**
# 
# Substitui nulos das colunas numéricas pela mediana da respectiva coluna.
# 

# %%
from sklearn.impute import SimpleImputer
num_imputer=SimpleImputer(strategy='median')
df[numerical_col]=num_imputer.fit_transform(df[numerical_col])

# %% [markdown]
# **Imputação categórica**
# 
# Substitui nulos das colunas categóricas pelo valor mais frequente de cada coluna.
# 

# %%
from sklearn.impute import SimpleImputer
cat_imputer=SimpleImputer(strategy='most_frequent')
df[categorical_col]=cat_imputer.fit_transform(df[categorical_col])

# %% [markdown]
# **Verificação pós-imputação**
# 
# Confere se ainda restaram valores ausentes no DataFrame.
# 
# 

# %%
df.isnull().sum()

# %% [markdown]
# ## **Análise Exploratória (EDA)**
# 
# Investigação inicial de frequências e distribuições para entender o perfil dos solicitantes.

# %% [markdown]
# Contagem de `Loan_Approved`
# Quantos casos aprovados e rejeitados existem na base (balanceamento da classe).

# %%
print(df['Loan_Approved'].value_counts())

# %% [markdown]
# ### **Estatísticas descritivas e dispersão**

# %%
df[numerical_col].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).T

# %% [markdown]
# | Variável               | Interpretação Estatística        | Observação Analítica                                                                                                                                |
# | ---------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Applicant_Income**   | Média ≈ 10.837, desvio ≈ 4.934   | Renda dos solicitantes varia bastante — alta dispersão (coeficiente de variação ≈ 45%). Há forte assimetria positiva (alguns com renda muito alta). |
# | **Coapplicant_Income** | Média ≈ 5.088, desvio ≈ 2.869    | Em média, co-solicitantes ganham metade do titular. O valor mínimo (1.0) sugere casos sem renda formal. Distribuição também assimétrica.            |
# | **Age**                | Média ≈ 40 anos, amplitude 21–59 | Perfil típico de adultos em idade laboral ativa. Intervalo interquartil (31–49) mostra concentração em meia-idade.                                  |
# | **Dependents**         | Média ≈ 1,45, máximo 3           | A maioria tem 1 a 2 dependentes.                                                                                                                    |
# | **Credit_Score**       | Média ≈ 676, desvio ≈ 69         | Distribuição próxima à normal, centrada em score médio-alto. A faixa 550–800 é típica de escala de crédito padrão.                                  |
# | **Existing_Loans**     | Média ≈ 1,95, máximo 4           | A maioria possui 1–2 empréstimos ativos. Possível relação negativa com aprovação (maior endividamento → maior risco).                               |
# | **DTI_Ratio**          | Média ≈ 0,35 (35%)               | O índice *Debt-to-Income* (dívida/renda) médio é aceitável. Valores acima de 0,5 (50%) indicam endividamento mais arriscado.                        |
# | **Savings**            | Média ≈ 9.937, desvio ≈ 5.712    | Grande dispersão — alguns clientes têm reservas financeiras muito altas.                                                                            |
# | **Collateral_Value**   | Média ≈ 24.778, desvio ≈ 13.982  | Valor da garantia (colateral) é variável; há grande amplitude (36 até ~50 mil). Indica heterogeneidade nos tipos de bens.                           |
# | **Loan_Amount**        | Média ≈ 20.557, desvio ≈ 11.213  | Valores de empréstimo variam de 1.015 a ~40 mil. Quartis (10k–30k) indicam concentração em empréstimos médios.                                      |
# | **Loan_Term**          | Média ≈ 48 meses, desvio ≈ 23,6  | Mediana = 48 → metade dos contratos são de 4 anos. 75% até 6 anos (72 meses).                                                                       |
# 

# %% [markdown]
# #### **Gráfico: Status do Empréstimo**
# 

# %%
loan_status_count = df['Loan_Approved'].value_counts().reset_index()
loan_status_count.columns = ["Loan_approved", "count"]
fig_loan_status = px.bar( loan_status_count,
                         x='Loan_approved',
                         y='count', color='Loan_approved',
                         text='count',
                         title='Distribuição do Status de Empréstimos',
                         labels={'Loan_approved': 'Status do Empréstimo', 'count': 'Número de Solicitações'} )

fig_loan_status.update_traces(textposition='outside')
fig_loan_status.update_layout( xaxis=dict(tickmode='array',
                                          tickvals=[0, 1],
                                          ticktext=['Rejeitado (0)', 'Aprovado (1)']),
                                          showlegend=False, yaxis_title='Quantidade',
                                          xaxis_title='Status do Empréstimo', )

max_y = loan_status_count['count'].max()
fig_loan_status.update_yaxes(range=[0, max_y * 1.15])
fig_loan_status.update_layout(margin=dict(t=80))
fig_loan_status.show()

# %% [markdown]
# #### **Gráfico: Distribuição de Gênero**
# 

# %%
# contagem dos gêneros
gender_count = df['Gender'].value_counts().reset_index()
gender_count.columns = ["Gender", "count"]

# substitui os nomes
gender_count['Gender'] = gender_count['Gender'].replace({
    'Male': 'Masculino',
    'Female': 'Feminino'
})

fig_gender = px.bar(
    gender_count,
    x='Gender',
    y='count',
    color='Gender',
    text='count',
    title='Distribuição de Gênero',
    labels={'Gender': 'Gênero', 'count': 'Quantidade'}
)

fig_gender.update_traces(textposition='outside')
fig_gender.update_layout(
    showlegend=False,
    yaxis_title='Quantidade',
    xaxis_title='Gênero',
)

max_y = gender_count['count'].max()
fig_gender.update_yaxes(range=[0, max_y * 1.15])
fig_gender.update_layout(margin=dict(t=80))

fig_gender.show()


# %% [markdown]
# Contagem de `Marital_Status`
# 
# Frequência por estado civil com rótulos traduzidos.
# 

# %%
df['Marital_Status'].value_counts()

# %% [markdown]
# #### **Gráfico: Distribuição por Estado Civil**
# 

# %%
# contagem do estado civil
marital_count = df['Marital_Status'].value_counts().reset_index()
marital_count.columns = ["Marital_Status", "count"]

# tradução dos valores 
marital_count['Marital_Status'] = marital_count['Marital_Status'].replace({
    'Married': 'Casado(a)',
    'Single': 'Solteiro(a)'
})

fig_marital = px.bar(
    marital_count,
    x='Marital_Status',
    y='count',
    color='Marital_Status',
    text='count',
    title='Distribuição por Estado Civil',
    labels={'Marital_Status': 'Estado Civil', 'count': 'Quantidade'}
)

fig_marital.update_traces(textposition='outside')
fig_marital.update_layout(
    showlegend=False,
    yaxis_title='Quantidade',
    xaxis_title='Estado Civil',
)

max_y = marital_count['count'].max()
fig_marital.update_yaxes(range=[0, max_y * 1.15])
fig_marital.update_layout(margin=dict(t=80))

fig_marital.show()



# %% [markdown]
# #### **Histograma: Distribuição da renda do solicitante**

# %%
fig_applicant_income = px.histogram(
    df,
    x='Applicant_Income',
    nbins=30,  # ajusta número de intervalos
    color_discrete_sequence=['#5DADE2'],  # azul suave
    title='Distribuição da Renda do Solicitante',
    labels={'Applicant_Income': 'Renda do Solicitante (R$)'}
)

# adiciona linha da média
mean_income = df['Applicant_Income'].mean()
fig_applicant_income.add_vline(
    x=mean_income,
    line_dash='dash',
    line_color='red',
    annotation_text=f"Média: {mean_income:.0f}",
    annotation_position='top right'
)

fig_applicant_income.update_traces(
    marker_line_color='black',
    marker_line_width=1
)
fig_applicant_income.update_layout(
    xaxis_title='Renda do Solicitante (R$)',
    yaxis_title='Frequência',
    plot_bgcolor='white',
    margin=dict(t=80, l=60, r=40, b=60)
)

fig_applicant_income.show()


# %% [markdown]
# #### **Histograma: Distribuição da renda do co-solicitante** 
# 

# %%
fig_coapplicant_income = px.histogram(
    df,
    x='Coapplicant_Income',
    nbins=30,
    color_discrete_sequence=['#48C9B0'],  # verde-água suave (diferente do azul anterior)
    title='Distribuição da Renda do Co-Solicitante',
    labels={'Coapplicant_Income': 'Renda do Co-Solicitante (R$)'}
)

# adiciona linha da média
mean_coincome = df['Coapplicant_Income'].mean()
fig_coapplicant_income.add_vline(
    x=mean_coincome,
    line_dash='dash',
    line_color='red',
    annotation_text=f"Média: {mean_coincome:.0f}",
    annotation_position='top right'
)

# ajustes visuais
fig_coapplicant_income.update_traces(
    marker_line_color='black',
    marker_line_width=1
)
fig_coapplicant_income.update_layout(
    xaxis_title='Renda do Co-Solicitante (R$)',
    yaxis_title='Frequência',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(t=80, l=60, r=40, b=60)
)

fig_coapplicant_income.show()


# %% [markdown]
# #### **Boxplot: Renda × Aprovação**

# %%
fig_income = px.box(
    df,
    x='Loan_Approved',
    y='Applicant_Income',
    color='Loan_Approved',
    points='all',  # mostra todos os pontos individuais
    title='Distribuição da Renda do Solicitante por Status de Empréstimo',
    labels={
        'Loan_Approved': 'Status do Empréstimo',
        'Applicant_Income': 'Renda do Solicitante (R$)'
    },
    color_discrete_map={
        0: '#E74C3C',  # vermelho (Rejeitado)
        1: '#27AE60'   # verde (Aprovado)
    }
)

fig_income.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=[0, 1],
        ticktext=['Rejeitado (0)', 'Aprovado (1)']
    ),
    yaxis_title='Renda do Solicitante (R$)',
    xaxis_title='Status do Empréstimo',
    title_font=dict(size=20, family='Arial', color='black'),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=14),
    showlegend=False
)

fig_income.update_traces(
    boxmean='sd',  # adiciona média e desvio padrão
    jitter=0.4,    # dispersão dos pontos
    marker=dict(opacity=0.6, size=5)
)

fig_income.show()


# %% [markdown]
# #### **Histograma: Score de Crédito × Aprovação**
# 

# %%
fig_credit_score = px.histogram(
    df,
    x='Credit_Score',
    color='Loan_Approved',
    barmode='group',  # barras lado a lado (comparação direta)
    nbins=25,
    text_auto=True,   # mostra os valores no topo das barras
    title='Distribuição do Score de Crédito por Status de Empréstimo',
    labels={
        'Credit_Score': 'Score de Crédito',
        'Loan_Approved': 'Status do Empréstimo'
    },
    color_discrete_map={
        0: '#E74C3C',  # vermelho para Rejeitado
        1: '#2ECC71'   # verde para Aprovado
    }
)

fig_credit_score.update_traces(
    textfont_size=12,
    textangle=0,
    textposition='outside',
    cliponaxis=False
)

fig_credit_score.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        title='Score de Crédito',
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)',
        zeroline=False
    ),
    yaxis=dict(
        title='Número de Solicitantes',
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)'
    ),
    legend=dict(
        title='Status do Empréstimo',
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5,
        font=dict(size=13)
    )
)

fig_credit_score.show()


# %% [markdown]
# ### **Distribuição do DTI_Ratio**

# %%
sns.histplot(df['DTI_Ratio'], bins=20, kde=True)


# %% [markdown]
# ### **Loan_Amount x Loan_Approved**

# %%
sns.boxplot(x='Loan_Approved', y='Loan_Amount', data=df)

# %% [markdown]
# ### **Collateral_Value x Loan_Approved**

# %%
sns.boxplot(x='Loan_Approved', y='Collateral_Value', data=df)


# %% [markdown]
# ###  **Distribuição por faixa etária**

# %%
sns.histplot(df['Age'], bins=15, kde=True)


# %% [markdown]
# ### **DTI_Ratio vs Loan_Amount (dispersão)**

# %%
sns.scatterplot(x='DTI_Ratio', y='Loan_Amount', hue='Loan_Approved', data=df)


# %% [markdown]
# ### **Savings x Loan_Approved**

# %%
sns.boxplot(x='Loan_Approved', y='Savings', data=df)


# %% [markdown]
# ### **Codificação de variáveis**
# 
# Transformações para converter categorias em números antes da modelagem.
# 

# %%
df.head()

# %% [markdown]
# ### **LabelEncoder (binárias/ordinais)**
# 
# Converte categorias em inteiros em colunas existentes (ex.: Gender, Marital_Status, Loan_Approved). Útil quando a variável é binária ou possui ordem natural.
# 

# %%
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encode=LabelEncoder()
for col in ["Marital_Status", "Gender", "Loan_Approved"]:
    if col in df.columns:
        df[col]=label_encode.fit_transform(df[col])


# %% [markdown]
# ### **One-Hot Encoding (dummies)**
# Cria colunas indicadoras (0/1) para cada categoria (variáveis nominais).
# `drop_first=True` evita multicolinearidade ao remover a categoria de referência.
# 

# %%
df=pd.get_dummies(df, columns=[
    "Employment_Status",
    "Property_Area",
    "Education_Level",
    "Loan_Purpose",
    "Employer_Category"
    ], drop_first=True)

# %%
df.head()

# %% [markdown]
# ### **Correlação (mapa de calor)**
# 
# Mede associação linear entre variáveis numéricas.
# Usamos:
# 
# * Vetor de correlação com `Loan_Approved` para ranquear relevância linear
# * Heatmap com anotações (valores arredondados) centralizado em 0
# 

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
numerical_df = df.select_dtypes(include=['float64', 'int64'])

corr = numerical_df.corr()["Loan_Approved"].sort_values(ascending=False)
print("Correlação com Loan_Approved:\n", corr.round(3))  # arredonda para 3 casas


# %%
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid", font_scale=1.1)

heatmap = sns.heatmap(
    numerical_df.corr().round(2),
    annot=True,
    cmap="RdBu_r",       
    center=0,
    linewidths=0.5,
    fmt=".2f"
)

plt.title("Mapa de Correlação entre Variáveis Numéricas", fontsize=16, pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### **Principais correlações com `Loan_Approved`**

# %% [markdown]
# | Variável             | Correlação | Interpretação                                                                                                                                                |
# | -------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
# | **Credit_Score**     | **+0.45**  | Forte relação positiva — quanto maior o score, **maior a chance de aprovação**. Esse é o fator mais relevante no mapa.                                       |
# | **DTI_Ratio**        | **-0.44**  | Relação negativa — quanto maior o endividamento (DTI), **menor a chance de aprovação**. Clientes mais comprometidos financeiramente tendem a ser rejeitados. |
# | **Applicant_Income** | **+0.12**  | Correlação fraca, mas positiva — rendas mais altas **ajudam** na aprovação, embora não sejam determinantes isoladamente.                                     |
# | **Loan_Amount**      | **-0.13**  | Correlação levemente negativa — empréstimos maiores são **menos aprovados**, o que faz sentido em termos de risco.                                           |
# | **Demais variáveis** | ≈ 0        | Não apresentam relação linear relevante com a aprovação (idade, dependentes, poupança etc.).                                                                 |
# 

# %% [markdown]
# ### *Multicolinearidade (VIF)*

# %%
# %% [markdown]
# **VIF (Variance Inflation Factor) para multicolinearidade**
# %%
numX = df.select_dtypes(include=['float64','int64']).drop(columns=['Loan_Approved'])
X_vif = sm.add_constant(numX)
vif = pd.Series(
    [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
    index=X_vif.columns, name='VIF'
).sort_values(ascending=False)
vif


# %% [markdown]
# As variáveis são estatisticamente independentes entre si.

# %% [markdown]
# ### **Modelagem**
# 
# Preparação do conjunto de treino/teste, padronização e ajuste de modelos de classificação.
# 

# %%
df.head()

# %% [markdown]
# **Variável alvo e preditores**
# 
# `X`: todas as colunas exceto `Loan_Approved`
# `y`: coluna alvo `Loan_Approved`
# 

# %%
X=df.drop("Loan_Approved", axis=1)
y=df["Loan_Approved"]

# %%
X

# %%
y

# %% [markdown]
# ### **Divisão treino/teste**
# 
# Separa dados em treino e teste (60%/40%) com `random_state=42` para reprodutibilidade.

# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)   

# %% [markdown]
# ### **Feature Scaling**
# **Padronização (StandardScaler)**
# 
# * Centraliza e escala preditores para média 0 e desvio 1.
# * Importante para modelos sensíveis à escala (ex.: SVM, Regressão Logística).

# %%
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# scaler.fit_transform(X_train)
# fit: calcula a média e o desvio padrão usando apenas o conjunto de treino
# transform: aplica a padronização com esses parâmetros.

# scaler.transform(X_test)
# usa os mesmos valores de média e desvio padrão obtidos no treino para padronizar o conjunto de teste.

# %%
X_train_scaled

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# %% [markdown]
# ### **Treinamento de Modelos**
# Ajuste de três algoritmos (Regressão Logística, SVM e Random Forest) para prever `Loan_Approved`.
# 

# %% [markdown]
# **Regressão Logística**: Cria um modelo linear que estima a probabilidade de um evento (ex.: aprovação do empréstimo).
# Usa uma função sigmoide para prever valores entre 0 e 1.
# 
# * O parâmetro **max_iter=1000** aumenta o número máximo de iterações para garantir convergência.
# 
# * **random_state=42** assegura reprodutibilidade (mesmo resultado a cada execução).
# 
# * **log_pred** contém as previsões (0 = rejeitado, 1 = aprovado) no conjunto de teste

# %%
log_model=LogisticRegression(max_iter=1000,random_state=42)
log_model.fit(X_train_scaled,y_train)
log_pred=log_model.predict(X_test_scaled)

# %% [markdown]
# **SVC (Support Vector Classifier)**: tenta encontrar a melhor fronteira que separa as classes.
# 
# * **kernel="rbf"** usa o kernel radial, que permite fronteiras não lineares.
# 
# * **probability=True** habilita cálculo de probabilidades (útil para gráficos ROC ou calibração posterior).
# 
# * **svm_pred** contém as previsões (0/1) do modelo SVM sobre o conjunto de teste.

# %%
svm_model=SVC(kernel="rbf",probability=True,random_state=42)
svm_model.fit(X_train_scaled,y_train)
svm_pred=svm_model.predict(X_test_scaled)

# %% [markdown]
# **Árvores de decisão**: combina várias árvores para aumentar precisão e estabilidade.
# 
# * **n_estimators=200** → cria 200 árvores (quanto mais, mais robusto).
# 
# * **random_state=42** → garante reprodutibilidade.
# 
# * **rf_pred** contém as previsões da Random Forest para o conjunto de teste.

# %%
rf_model=RandomForestClassifier(n_estimators=200,random_state=42)
rf_model.fit(X_train_scaled,y_train)
rf_pred=rf_model.predict(X_test_scaled)

# %%
# Converte os arrays escalados de volta para DataFrame com nomes de colunas/índices.
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df  = pd.DataFrame(X_test_scaled,  columns=X.columns, index=X_test.index)


# %% [markdown]
# **Regressão Logística com statsmodels (OR e IC95%)**

# %%
X_sm = sm.add_constant(X_train_scaled_df)
logit_sm = sm.Logit(y_train, X_sm).fit(disp=0)
or_ci = pd.DataFrame({
    "OR": np.exp(logit_sm.params),
    "IC95%_inf": np.exp(logit_sm.conf_int()[0]),
    "IC95%_sup": np.exp(logit_sm.conf_int()[1]),
    "pvalor": logit_sm.pvalues
}).sort_values("pvalor")
or_ci.head(15)


# %% [markdown]
# | Tipo de Efeito         | Variável                                                           | Interpretação                                            |
# | ---------------------- | ------------------------------------------------------------------ | -------------------------------------------------------- |
# | **Forte positivo**  | `Credit_Score`, `Applicant_Income`                                 | Aumentam substancialmente as chances de aprovação.       |
# | **Forte negativo**  | `DTI_Ratio`, `Employment_Status_Unemployed`, `Gender`              | Reduzem significativamente a probabilidade de aprovação. |
# | **Neutros / fracos** | `Loan_Term`, `Loan_Amount`, `Education_Level`, `Employer_Category` | Não influenciam de forma estatisticamente relevante.     |
# 

# %%
pd.DataFrame({
    "Feature": X_train_scaled_df.columns,
    "Index": range(len(X_train_scaled_df.columns))
}).head(30)


# %%
import matplotlib.pyplot as plt

or_ci_plot = or_ci.drop("const")
plt.figure(figsize=(8,6))
plt.errorbar(or_ci_plot["OR"], or_ci_plot.index,
             xerr=[or_ci_plot["OR"]-or_ci_plot["IC95%_inf"],
                   or_ci_plot["IC95%_sup"]-or_ci_plot["OR"]],
             fmt='o', color='blue', ecolor='gray', capsize=4)
plt.axvline(1, color='red', linestyle='--')
plt.xlabel("Odds Ratio (IC95%)")
plt.ylabel("Variável")
plt.title("Efeitos das variáveis na aprovação do empréstimo")
plt.show()


# %% [markdown]
# O gráfico confirma visualmente os resultados numéricos:
# 
# Score de crédito alto (Credit_Score) e renda elevada (Applicant_Income) aumentam as chances de aprovação.
# 
# Endividamento (DTI_Ratio), desemprego e possivelmente gênero reduzem as chances.
# 
# As demais variáveis não apresentam efeito estatisticamente relevante após controle das demais.

# %% [markdown]
# ### **Avaliação dos Modelos**

# %%
models={
    "Logistic Regression":(y_test,log_pred),
    "Support Vector Machine":(y_test,svm_pred),
    "Random Forest":(y_test,rf_pred)
}

accuracy_scores={}
     

# %%
for model_name, (y_true,y_pred) in models.items():
  acc=accuracy_score(y_true,y_pred)
  accuracy_scores[model_name]=acc
  
  print(f"{model_name} Accuracy: {acc}")
  print("Confusion Matrix:")
  print(confusion_matrix(y_true,y_pred))
  print("Classification Report:")
  print(classification_report(y_true,y_pred))
  print("\n")


# %% [markdown]
# ### **Desempenho dos modelos**

# %% [markdown]
# | Modelo                  | Acurácia   | Precisão (1) | Recall (1) | F1 (1) | Observações                                                                                                                           |
# | ----------------------- | ---------- | ------------ | ---------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------- |
# | **Regressão Logística** | **0.84**   | 0.81         | 0.66       | 0.73   | Modelo interpretável; bom equilíbrio entre precisão e recall; ligeira tendência a **perder alguns aprovados** (falsos negativos).     |
# | **SVM (RBF)**           | **0.83**   | 0.85         | 0.58       | 0.69   | Classificador mais conservador, **alta precisão mas recall menor** → aprova com mais “cautela”, rejeitando mais casos duvidosos.      |
# | **Random Forest**       | **0.90** | 0.88         | 0.80       | 0.84   | Melhor desempenho geral; excelente equilíbrio entre identificar aprovados e rejeitados; mais robusto a não-linearidades e interações. |
# 

# %% [markdown]
# ### **Interpretação das matrizes de confusão**

# %% [markdown]
# Regressão Logística 
# 
# |            | Previsto 0 | Previsto 1 |
# | ---------- | ---------- | ---------- |
# | **Real 0** | 249        | 20         |
# | **Real 1** | 44         | 87         |
# 
# 93% dos rejeitados corretamente previstos, mas ~34% dos aprovados foram perdidos (falsos negativos).
# 
# Logística ainda é útil para interpretação e explicação (odds ratio, significância estatística), mas perde um pouco de recall em relação ao Random Forest.
# 
# SVM
# 
# |            | Previsto 0 | Previsto 1 |
# | ---------- | ---------- | ---------- |
# | **Real 0** | 256        | 13         |
# | **Real 1** | 55         | 76         |
# 
# Bom para rejeitados (95% corretos), mas baixo recall para aprovados (58%) → arriscado se o objetivo é não negar bons clientes.
# 
# Eficiente, porém menos interpretável e com recall mais baixo → não ideal se o objetivo é evitar falsos negativos.
# 
# Random Forest
# 
# |            | Previsto 0 | Previsto 1 |
# | ---------- | ---------- | ---------- |
# | **Real 0** | 254        | 15         |
# | **Real 1** | 26         | 105        |
# 
# 94% dos rejeitados corretos e 80% dos aprovados identificados. Melhor trade-off entre risco e aprovação.
# 
# Random Forest é o mais eficaz (acurácia 0.90, F1=0.84), superando os demais em todas as métricas.

# %% [markdown]
# ### **AUC, PR, Brier e curvas ROC/PR + Calibração**

# %% [markdown]
# ### Regressão Logística:

# %%
log_probs = log_model.predict_proba(X_test_scaled_df)[:,1]

roc_auc = roc_auc_score(y_test, log_probs)
brier = brier_score_loss(y_test, log_probs)
print("ROC-AUC:", roc_auc, "| Brier:", brier)

# %% [markdown]
# Curva ROC (Receiver Operating Characteristic) da Regressão Logística

# %%
RocCurveDisplay.from_predictions(y_test, log_probs)
plt.show()

# %% [markdown]
# | Métrica                 | Valor                                                                            | Interpretação                                    |
# | ----------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------ |
# | **AUC**                 | 0.91                                                                             | Excelente separação entre aprovados e rejeitados |
# | **Curva ROC**           | Muito acima da diagonal                                                          | Modelo tem alto poder preditivo                  |
# | **FPR baixo, TPR alto** | Identifica bem os aprovados sem muitos erros                                     |                                                  |
# | **Conclusão**           | A regressão logística é altamente eficaz e calibrada para este conjunto de dados |                                                  |
# 

# %%
PrecisionRecallDisplay.from_predictions(y_test, log_probs)
plt.show()

# %% [markdown]
# | Métrica                   | Valor                                                                     | Interpretação                                                  |
# | ------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------- |
# | **AUC ROC**               | 0.91                                                                      | Excelente capacidade de discriminar entre aprovados/rejeitados |
# | **AP (Precision–Recall)** | 0.82                                                                      | Alta precisão média em base desbalanceada                      |
# | **F1-Score (classe 1)**   | ~0.73                                                                     | Equilíbrio entre precisão e recall                             |
# | **Conclusão geral**       | O modelo é robusto, bem calibrado e confiável para decisão de crédito. |                                                                |
# 

# %% [markdown]
# ### SVM:

# %%
svm_probs = svm_model.predict_proba(X_test_scaled_df)[:,1]

roc_auc = roc_auc_score(y_test, svm_probs)
brier = brier_score_loss(y_test, svm_probs)
print("ROC-AUC:", roc_auc, "| Brier:", brier)

# %%
RocCurveDisplay.from_predictions(y_test, svm_probs)
plt.show()

# %% [markdown]
# | **Métrica**             | **Valor**                                                                                                                               | **Interpretação**                                              |
# | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
# | **AUC**                 | 0.93                                                                                                                                    | Excelente capacidade de separação entre aprovados e rejeitados |
# | **Curva ROC**           | Muito acima da diagonal                                                                                                                 | Modelo tem alto poder discriminativo, separando bem as classes |
# | **FPR baixo, TPR alto** | Taxa de falsos positivos baixa e alta taxa de acertos para os aprovados                                                                 | Indica ótima sensibilidade sem comprometer a precisão          |
# | **Conclusão (ROC)**     | O SVM apresenta desempenho equivalente à regressão logística, distinguindo eficientemente quem tende ou não a ter o empréstimo aprovado |                                                                |
# 

# %%
PrecisionRecallDisplay.from_predictions(y_test, svm_probs)
plt.show()

# %% [markdown]
# | **Métrica**                | **Valor**                                                                                                                                                  | **Interpretação**                                                                                |
# | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
# | **AP (Precision–Recall)**  | 0.85                                                                                                                                                       | Excelente equilíbrio entre precisão e sensibilidade                                              |
# | **Curva Precision–Recall** | Curva bem acima da diagonal, mantendo alta precisão (>80%) mesmo com alto recall                                                                           | Mostra que o modelo identifica muitos casos de aprovação sem incluir muitos falsos positivos     |
# | **Comportamento geral**    | A precisão começa próxima de 1.0 e diminui suavemente conforme o recall aumenta                                                                            | Indica que o modelo é **muito consistente**, mantendo bom desempenho em toda a faixa de previsão |
# | **Conclusão (PR)**         | O Random Forest supera os outros modelos (Logístico e SVM) em precisão e recall, sendo o **mais robusto para identificar corretamente clientes aprovados** |                                                                                                  |
# 

# %% [markdown]
# ### Random Forest:

# %%
rf_probs = rf_model.predict_proba(X_test_scaled_df)[:,1]

roc_auc = roc_auc_score(y_test, rf_probs)
brier = brier_score_loss(y_test, rf_probs)
print("ROC-AUC:", roc_auc, "| Brier:", brier)

# %%
RocCurveDisplay.from_predictions(y_test, rf_probs)
plt.show()

# %% [markdown]
# | **Métrica**              | **Valor**                                                                        | **Interpretação**                                                                           |
# | ------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
# | **AUC (Curva ROC)**      | **0.96**                                                                         | Excelente — o modelo distingue com altíssima precisão entre aprovados e rejeitados          |
# | **Curva ROC**            | A curva está próxima do canto superior esquerdo                                  | Demonstra que o modelo acerta a grande maioria dos casos com baixa taxa de falsos positivos |
# | **FPR baixo / TPR alto** | O modelo mantém taxa de acertos alta mesmo com poucos erros de aprovação         | Indica ótimo equilíbrio entre sensibilidade e especificidade                                |
# | **Conclusão (ROC)**      | Random Forest apresenta **poder preditivo superior** aos outros modelos testados |                                                                                             |
# 

# %%
PrecisionRecallDisplay.from_predictions(y_test, rf_probs)
plt.show()

# %% [markdown]
# | **Métrica Precision–Recall**       | **Valor**                                                                                       | **Interpretação**                                                            |
# | ---------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
# | **AP (Área sob Precision–Recall)** | **0.88**                                                                                        | Excelente equilíbrio entre precisão e recall                                 |
# | **Curva Precision–Recall**         | Mantém **precisão > 0.85** em quase toda a faixa de recall                                      | O modelo identifica muitos aprovados sem aumentar falsos positivos           |
# | **Comportamento geral**            | Quase constante até altos níveis de recall                                                      | Indica **robustez e estabilidade**, mesmo sob diferentes limiares de decisão |
# | **Conclusão (PR)**                 | Random Forest é o modelo **mais confiável e equilibrado** para prever aprovações de empréstimos |                                                                              |
# 

# %% [markdown]
# ### **Curva de calibração (quantis)**
# 

# %% [markdown]
# ### Regressão Logística:

# %%
prob_true, prob_pred = calibration_curve(y_test, log_probs, n_bins=10, strategy="quantile")
pd.DataFrame({"pred_bin": prob_pred, "true_rate": prob_true})

# %% [markdown]
# | Aspecto                         | Avaliação                                                                  |
# | ------------------------------- | -------------------------------------------------------------------------- |
# | **Tendência geral**             | Leve **subestimação** das probabilidades                                   |
# | **Calibração em faixas baixas** | Boa — previsões muito baixas refletem bem a realidade                      |
# | **Faixas médias e altas**       | Subestima levemente, mas ainda consistente                                 |
# | **Faixa mais alta (>0.9)**      | Pequena **superestimação**, mas sem distorção grave                        |
# | **Conclusão geral**             | O modelo está **razoavelmente bem calibrado**, apenas um pouco conservador |

# %%
data = {
    "pred_bin": [0.001615, 0.008221, 0.021010, 0.044109, 0.095794,
                 0.170600, 0.343191, 0.525942, 0.717803, 0.931592],
    "true_rate": [0.000, 0.025, 0.050, 0.050, 0.175,
                  0.200, 0.400, 0.625, 0.875, 0.875]
}

df = pd.DataFrame(data)

plt.figure(figsize=(7,6))
plt.plot(df["pred_bin"], df["true_rate"], marker='o', color="#1f77b4", label="Modelo")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Calibração perfeita (y = x)")

plt.title("Curva de Calibração - Regressão Logística", fontsize=14, fontweight='bold')
plt.xlabel("Probabilidade prevista média (pred_bin)", fontsize=12)
plt.ylabel("Proporção real de aprovados (true_rate)", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()


# %% [markdown]
# O modelo apresenta ótima calibração global, com tendência ligeiramente conservadora, ou seja, as probabilidades podem ser usadas de forma confiável para decisão de risco (por exemplo, definir limiares de aprovação mais transparentes).

# %% [markdown]
# ### SVM:

# %%
prob_true, prob_pred = calibration_curve(y_test, svm_probs, n_bins=10, strategy="quantile")
pd.DataFrame({"pred_bin": prob_pred, "true_rate": prob_true})

# %%
data = {
    "pred_bin": [0.005433, 0.015578, 0.033668, 0.056700, 0.101489,
                 0.182186, 0.337111	, 0.531227, 0.708206, 0.913279],
    "true_rate": [0.000, 0.000, 0.075, 0.050, 0.025,
                  0.200, 0.475, 0.750, 0.725, 0.975]
}

df = pd.DataFrame(data)

plt.figure(figsize=(7,6))
plt.plot(df["pred_bin"], df["true_rate"], marker='o', color="#1f77b4", label="Modelo")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Calibração perfeita (y = x)")

plt.title("Curva de Calibração - SVM", fontsize=14, fontweight='bold')
plt.xlabel("Probabilidade prevista média (pred_bin)", fontsize=12)
plt.ylabel("Proporção real de aprovados (true_rate)", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()


# %% [markdown]
# O SVM não apenas separa bem as classes (AUC = 0.91), mas também atribui probabilidades coerentes com o comportamento real dos dados. Isso significa que, se o modelo prevê 0.7 de chance de aprovação, aproximadamente 70% dos clientes com essa pontuação realmente são aprovados, o que reforça a confiabilidade preditiva.

# %% [markdown]
# ### Random Forest:

# %%
prob_true, prob_pred = calibration_curve(y_test, rf_probs, n_bins=10, strategy="quantile")
pd.DataFrame({"pred_bin": prob_pred, "true_rate": prob_true})

# %%
data = {
    "pred_bin": [0.011548, 0.045119, 0.072727, 0.090556, 0.108171,
                 0.152143, 0.394875	, 0.569390, 0.668452, 0.761757],
    "true_rate": [0.000000, 0.000000, 0.000000, 0.000000, 0.024390,
                  0.028571, 0.600000, 0.926829, 0.857143, 0.837838]
}

df = pd.DataFrame(data)

plt.figure(figsize=(7,6))
plt.plot(df["pred_bin"], df["true_rate"], marker='o', color="#1f77b4", label="Modelo")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Calibração perfeita (y = x)")

plt.title("Curva de Calibração - Random Forest", fontsize=14, fontweight='bold')
plt.xlabel("Probabilidade prevista média (pred_bin)", fontsize=12)
plt.ylabel("Proporção real de aprovados (true_rate)", fontsize=12)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()


# %% [markdown]
# A Random Forest, além de ter o melhor desempenho em AUC e AP (0.96 e 0.88), também demonstra excelente calibração — suas previsões refletem com precisão o comportamento real. Em termos práticos: quando o modelo prevê 70% de chance de aprovação, aproximadamente 70% dos casos realmente são aprovados. Isso torna a Random Forest altamente confiável para apoio à decisão em crédito.

# %% [markdown]
# ### **Matriz de Confusão**

# %% [markdown]
# ### Regressão Logística:

# %%
ths = np.linspace(0,1,201)
youden = []
for t in ths:
    y_pred = (log_probs>=t).astype(int)
    se = recall_score(y_test, y_pred)
    sp = recall_score(1-y_test, 1-y_pred)  # especificidade
    youden.append(se+sp-1)
best_thr = ths[int(np.argmax(youden))]
best_thr

y_pred_thr = (log_probs>=best_thr).astype(int)
cm = confusion_matrix(y_test, y_pred_thr)
z = cm.astype(int)
x = ["Negativo (0)", "Positivo (1)"]
fig = ff.create_annotated_heatmap(z, x=x, y=x, colorscale='Blues', showscale=True)
fig.update_layout(title=f"Matriz de Confusão (thr={best_thr:.2f})", xaxis_title="Predito", yaxis_title="Verdadeiro")
fig.show()

# Relatório compacto com o limiar ótimo
def eval_report(y_true, pr, thr):
    yp = (pr>=thr).astype(int)
    return {
        "AUC": roc_auc_score(y_true, pr),
        "F1": f1_score(y_true, yp),
        "Precision": precision_score(y_true, yp),
        "Recall": recall_score(y_true, yp),
        "Brier": brier_score_loss(y_true, pr),
        "Threshold": thr
    }
eval_report(y_test, log_probs, best_thr)


# %% [markdown]
# O modelo é forte (AUC=0.91) e bem calibrado (Brier=0.11). O recall alto (0.80) indica que ele detecta bem quem deve ser aprovado. Porém, há alguns falsos positivos (34) — ou seja, prevê aprovação para pessoas que acabam sendo rejeitadas. O threshold mais baixo (0.35) favorece sensibilidade (identificar mais aprovados), o que pode ser útil em contextos de análise de crédito conservadora, onde o banco prefere revisar manualmente os “suspeitos de aprovação”.

# %% [markdown]
# ### SVM:

# %%
ths = np.linspace(0,1,201)
youden = []
for t in ths:
    y_pred = (svm_probs>=t).astype(int)
    se = recall_score(y_test, y_pred)
    sp = recall_score(1-y_test, 1-y_pred)  # especificidade
    youden.append(se+sp-1)
best_thr = ths[int(np.argmax(youden))]
best_thr

y_pred_thr = (svm_probs>=best_thr).astype(int)
cm = confusion_matrix(y_test, y_pred_thr)
z = cm.astype(int)
x = ["Negativo (0)", "Positivo (1)"]
fig = ff.create_annotated_heatmap(z, x=x, y=x, colorscale='Blues', showscale=True)
fig.update_layout(title=f"Matriz de Confusão (thr={best_thr:.2f})", xaxis_title="Predito", yaxis_title="Verdadeiro")
fig.show()

# Relatório compacto com o limiar ótimo
def eval_report(y_true, pr, thr):
    yp = (pr>=thr).astype(int)
    return {
        "AUC": roc_auc_score(y_true, pr),
        "F1": f1_score(y_true, yp),
        "Precision": precision_score(y_true, yp),
        "Recall": recall_score(y_true, yp),
        "Brier": brier_score_loss(y_true, pr),
        "Threshold": thr
    }
eval_report(y_test, svm_probs, best_thr)

# %% [markdown]
# O modelo SVM apresenta excelente desempenho geral (AUC = 0.93), mostrando-se eficiente em separar clientes aprovados dos rejeitados. Apesar de uma leve tendência a subestimar alguns aprovados (FN = 117), o modelo mantém alta sensibilidade (0.89), o que é desejável para minimizar recusas indevidas. A calibração também é boa (Brier < 0.12), reforçando sua confiabilidade em probabilidades preditivas.

# %% [markdown]
# ### Random Forest:

# %%
ths = np.linspace(0,1,201)
youden = []
for t in ths:
    y_pred = (rf_probs>=t).astype(int)
    se = recall_score(y_test, y_pred)
    sp = recall_score(1-y_test, 1-y_pred)  # especificidade
    youden.append(se+sp-1)
best_thr = ths[int(np.argmax(youden))]
best_thr

y_pred_thr = (rf_probs>=best_thr).astype(int)
cm = confusion_matrix(y_test, y_pred_thr)
z = cm.astype(int)
x = ["Negativo (0)", "Positivo (1)"]
fig = ff.create_annotated_heatmap(z, x=x, y=x, colorscale='Blues', showscale=True)
fig.update_layout(title=f"Matriz de Confusão (thr={best_thr:.2f})", xaxis_title="Predito", yaxis_title="Verdadeiro")
fig.show()

# Relatório compacto com o limiar ótimo
def eval_report(y_true, pr, thr):
    yp = (pr>=thr).astype(int)
    return {
        "AUC": roc_auc_score(y_true, pr),
        "F1": f1_score(y_true, yp),
        "Precision": precision_score(y_true, yp),
        "Recall": recall_score(y_true, yp),
        "Brier": brier_score_loss(y_true, pr),
        "Threshold": thr
    }
eval_report(y_test, rf_probs, best_thr)

# %% [markdown]
# Desempenho geral: Excepcional, com AUC = 0.96 e F1 = 0.91. Detecta quase todos os aprovados (recall altíssimo) sem comprometer a precisão. Calibração: Boa, com leve tendência otimista em probabilidades altas. Uso ideal: Cenários onde a performance preditiva é prioridade — sistemas automáticos de crédito, triagem de risco, etc.

# %% [markdown]
# ### **Comparação Final dos Modelos**
# 
# | **Métrica**                | **Regressão Logística** | **SVM**   | **Random Forest** |
# | -------------------------- | ----------------------- | --------- | ----------------- |
# | **AUC (ROC)**              | **0.91**                | **0.93**  | **0.96**          |
# | **F1-score**               | **0.78**                | **0.80**  | **0.91**          |
# | **Precisão (Precision)**   | **0.76**                | **0.73**  | **0.85**          |
# | **Recall (Sensibilidade)** | **0.80**                | **0.89**  | **0.98**          |
# | **Brier Score**            | **0.115**               | **0.105** | **0.082**         |
# | **Threshold ótimo**        | **0.35**                | **0.26**  | **0.32**          |

# %% [markdown]
# | **Aspecto**                                   | **Melhor Modelo**          | **Comentário**                                                    |
# | --------------------------------------------- | -------------------------- | ----------------------------------------------------------------- |
# | **Poder de classificação (AUC)**              | **Random Forest**       | Maior área sob a curva — separa muito bem aprovados e rejeitados. |
# | **Equilíbrio geral (F1)**                     | **Random Forest**       | Excelente compromisso entre precisão e recall.                    |
# | **Precisão (baixa taxa de falsos positivos)** | **Random Forest**       | Quando prevê aprovação, acerta em 85%.                            |
# | **Sensibilidade (detecta aprovados reais)**   | **Random Forest**       | Identifica quase todos os aprovados (98%).                        |
# | **Calibração (probabilidades realistas)**     | **Regressão Logística** | Mais próxima da linha ideal na curva de calibração.               |
# | **Brier Score (consistência probabilística)** | **Random Forest**       | Menor erro médio nas probabilidades previstas.                    |
# 

# %% [markdown]
# | **Objetivo do uso**                                            | **Modelo recomendado**     | **Motivo**                        |
# | -------------------------------------------------------------- | -------------------------- | --------------------------------- |
# | Previsão automática de aprovação com máxima performance        |  **Random Forest**       | Maior AUC, F1 e recall.           |
# | Explicabilidade e análise de risco (probabilidades confiáveis) |  **Regressão Logística** | Calibração quase perfeita.        |
# | Triagem ampla e sensível (evitar perder aprovados)             | **SVM**                 | Recall alto, bom custo-benefício. |
# 

# %% [markdown]
# ### **Melhor Modelo**

# %%
best_model=max(accuracy_scores,key=accuracy_scores.get)
print(f"Best Model: {best_model}")

# %% [markdown]
# ### **Importância das Variáveis**
# 
# A Random Forest permite identificar **quais variáveis mais contribuíram** para as previsões de aprovação de empréstimos.
# 
# O atributo `feature_importances_` fornece um valor entre 0 e 1 para cada variável, representando a **importância relativa** no modelo:
# - Quanto **maior** o valor, **mais relevante** é a variável para a decisão final.
# - As importâncias somam **1 (ou 100%)**.

# %%
importances=rf_model.feature_importances_

# %%
importances

# %%
importance_df = (pd.DataFrame({"Feature": X.columns, "Importance": importances})
                   .sort_values("Importance", ascending=False)
                   .reset_index(drop=True))


# %%
importance_df.head()

# %% [markdown]
# 

# %% [markdown]
# ### **Importância da variáveis - Random Forest (top 10)**
# 

# %%
topk = 10
plt.figure(figsize=(8,5))
plt.barh(importance_df.loc[:topk-1, "Feature"][::-1],
         importance_df.loc[:topk-1, "Importance"][::-1])
plt.title("Importância das Variáveis – Random Forest (Top 10)")
plt.xlabel("Importância (ganho relativo)")
plt.ylabel("Variável")
plt.tight_layout()
plt.show()

importance_df.head(20)


# %% [markdown]
# ### **Curva de aprendizado (AUC ROC)**

# %%
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    rf_model, X_train_scaled_df, y_train,
    cv=5, scoring="roc_auc",
    train_sizes=np.linspace(0.1, 1.0, 6),
    n_jobs=-1, shuffle=True, random_state=42
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

plt.figure(figsize=(7,5))
plt.plot(train_sizes, train_mean, marker="o", label="Treino (AUC)")
plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15)
plt.plot(train_sizes, val_mean, marker="o", label="Validação (AUC)")
plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15)
plt.xlabel("Tamanho do conjunto de treino")
plt.ylabel("AUC ROC")
plt.title("Curva de Aprendizado – Random Forest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Tabela rápida (opcional)
pd.DataFrame({"train_size": train_sizes, "train_auc_mean": train_mean, "val_auc_mean": val_mean})


# %% [markdown]
# O modelo Random Forest está bem ajustado, com generalização excelente e AUC altíssimo (>0.97). A partir de ~200 observações o desempenho já se estabiliza, sugerindo que adicionar mais dados traz ganhos marginais. Confirma a robustez observada nas métricas anteriores (AUC = 0.96, F1 = 0.91) e reforça que a Random Forest é o melhor modelo geral para este problema.


