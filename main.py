import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 1000)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.float_format", "{:.2f}".format)

sns.set_theme(style="whitegrid")

# Create directory
os.makedirs('fig/exploratoria', exist_ok=True)

# Load data
try:
    df = pd.read_csv('dados/Crop_recommendation.csv')
except FileNotFoundError:
    # Just in case the file is named differently in the environment, but based on history it is 'Crop_recommendation.csv'
    df = pd.read_csv('Crop_recommendation.csv')

def perform_feature_engineering(data):
    df_eng = data.copy()
    df_eng['total_nutrients'] = df_eng['N'] + df_eng['P'] + df_eng['K']
    df_eng['N_P_ratio'] = df_eng['N'] / (df_eng['P'] + 1)
    df_eng['P_K_ratio'] = df_eng['P'] / (df_eng['K'] + 1)
    df_eng['temp_hum_interaction'] = df_eng['temperature'] * df_eng['humidity']
    df_eng['rain_ph_ratio'] = df_eng['rainfall'] / (df_eng['ph'] + 1)
    return df_eng

df_final = perform_feature_engineering(df)

mapa_colunas = {
    'N': 'Nitrogênio',
    'P': 'Fósforo',
    'K': 'Potássio',
    'temperature': 'Temperatura',
    'humidity': 'Umidade',
    'ph': 'pH_do_Solo',
    'rainfall': 'Precipitação',
    'label': 'Cultura',
    'total_nutrients': 'Total_de_Nutrientes',
    'N_P_ratio': 'Razão_N/P',
    'P_K_ratio': 'Razão_P/K',
    'temp_hum_interaction': 'Interação_Temp-Umidade',
    'rain_ph_ratio': 'Razão_Chuva/pH'
}

df_final = df_final.rename(columns=mapa_colunas)

# 2. Target Distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df_final, x='Cultura', palette='viridis', order=df_final['Cultura'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Distribuição das Culturas no Dataset')
plt.tight_layout()
plt.savefig('fig/exploratoria/target_distribution.png')

# 3. Correlation Matrix
plt.figure(figsize=(12, 10))
corr = df_final.drop('Cultura', axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.savefig('fig/exploratoria/correlation_matrix_eda.png')

# 4. Histograms
numeric_cols = df_final.select_dtypes(include=[np.number]).columns
n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df_final[col], kde=True, ax=axes[i], color='teal')
    axes[i].set_title(f'Distribuição de {col}')
    axes[i].set_xlabel('')
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig('fig/exploratoria/feature_histograms.png')

# 5. Boxplots - FIX IS HERE
key_features = ['Nitrogênio', 'Fósforo', 'Potássio', 'Temperatura', 'Umidade', 'Precipitação', 'Total_de_Nutrientes']
# Changed 'Total de Nutrientes' to 'Total_de_Nutrientes' to match the column name

fig, axes = plt.subplots(len(key_features), 1, figsize=(15, 6 * len(key_features)))
for i, col in enumerate(key_features):
    sns.boxplot(data=df_final, x='Cultura', y=col, ax=axes[i], palette='Set2')
    axes[i].set_title(f'Variação de {col} por Cultura')
    axes[i].tick_params(axis='x', rotation=90)
    axes[i].set_xlabel('')
plt.tight_layout()
plt.savefig('fig/exploratoria/boxplots_by_crop.png')

print("Code executed successfully.")
print(df_final.head())