import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Carregar o CSV
arquivo_csv = "base_dados_tr.csv"
df = pd.read_csv(arquivo_csv, sep=";")

# Verificar os tipos dos dados
print(df.dtypes)

# Converter a coluna de data para o formato correto
df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')

# Verificar os tipos dos dados
print(df.dtypes)

# Separar dados históricos (com TR preenchida) e dados futuros (TR vazia)
df_historico = df[df['tr'].notna()]
df_futuro = df[df['tr'].isna()]

# Definir as features e o target para treinamento
X_historico = df_historico[['selic', 'ipca', 'dolar']]
y_historico = df_historico['tr']

# Features para prever TR nos dados futuros
X_futuro = df_futuro[['selic', 'ipca', 'dolar']]

# Normalizar as features
scaler_X = StandardScaler()
X_historico_scaled = scaler_X.fit_transform(X_historico)
X_futuro_scaled = scaler_X.transform(X_futuro)

# Dividir os dados históricos em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_historico_scaled, y_historico, test_size=0.2, random_state=42)

# Criar e treinar o modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f'\nDesempenho do modelo Random Forest:')
print(f'R² Score: {r2:.3f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')

# Visualização: TR Real vs TR Prevista
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Previsões')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Linha Ideal')
plt.xlabel('TR Real')
plt.ylabel('TR Prevista')
plt.title('Comparação entre TR Real e Prevista')
plt.legend()
plt.grid(True)
plt.show()

#%% Importância das Variáveis - Random Forest
importances = rf_model.feature_importances_
feature_names = X_futuro.columns

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='green')
plt.xlabel('Importância')
plt.title('Importância das Features - Random Forest')
plt.show()

# Prever a TR para os dados futuros
tr_futuro_pred = rf_model.predict(X_futuro_scaled)

# Adicionar as previsões ao DataFrame
df.loc[df['tr'].isna(), 'tr'] = tr_futuro_pred

# Salvar o DataFrame atualizado com as previsões
df.to_csv("dados_tr_previstos_completos.csv", sep=";", decimal=",", index=False)

print("\nPrevisões da TR para os dados futuros foram salvas em 'dados_tr_previstos_completos.csv'")