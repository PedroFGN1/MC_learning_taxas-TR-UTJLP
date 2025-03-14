import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Carregar os dados
# Substitua 'caminho_arquivo.csv' pelo caminho do seu arquivo
# Usar separador ';' e converter vírgulas para pontos nos decimais
df = pd.read_csv('datas/tr(12m)1995-2025.csv', sep=';', decimal=',')

# Transformar os dados (desempilhar meses)
df = df.melt(id_vars='ano', var_name='mes', value_name='taxa')

# Criar coluna de data
# Convertendo mês para string e preenchendo com zero à esquerda
df['mes'] = df['mes'].astype(int).astype(str).str.zfill(2)
df['data'] = pd.to_datetime(df['ano'].astype(str) + '-' + df['mes'] + '-01')

# Definir data como índice
df.set_index('data', inplace=True)

# Ordenar por data
df = df.sort_index()

# Verificar valores ausentes ou infinitos
df['taxa'] = pd.to_numeric(df['taxa'], errors='coerce')
df.dropna(subset=['taxa'], inplace=True)

# Visualizar os dados
plt.figure(figsize=(10, 6))
plt.plot(df['taxa'], label='Taxa histórica')
plt.title('Série Temporal da Taxa Referencial')
plt.legend()
plt.show()

# Teste de estacionariedade (Dickey-Fuller)
result = adfuller(df['taxa'])
print(f'Estatística de Teste: {result[0]}')
print(f'P-valor: {result[1]}')
print('Valores Críticos:')
for key, value in result[4].items():
    print(f'   {key}: {value}')

if result[1] > 0.05:
    print("A série não é estacionária.")
else:
    print("A série é estacionária.")

# Decomposição da série temporal
decomposicao = seasonal_decompose(df['taxa'], model='additive', period=12)
decomposicao.plot()
plt.show()

# Definir parâmetros ARIMA (p, d, q)
p, d, q = 1, 1, 1  # Isso pode ser ajustado

# Dividir os dados em treino e teste
tamanho_treino = int(len(df) * 0.8)
treino, teste = df[:tamanho_treino], df[tamanho_treino:]

# Ajustar modelo ARIMA
modelo = ARIMA(treino['taxa'], order=(p, d, q))
modelo_ajustado = modelo.fit()

# Fazer previsões
previsoes = modelo_ajustado.forecast(steps=len(teste))

# Avaliar modelo
mse = mean_squared_error(teste['taxa'], previsoes)
rmse = np.sqrt(mse)
mae = mean_absolute_error(teste['taxa'], previsoes)
mape = np.mean(np.abs((teste['taxa'] - previsoes) / teste['taxa'])) * 100
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'MAPE: {mape}%')

# Visualizar previsões
plt.figure(figsize=(10, 6))
plt.plot(treino['taxa'], label='Treino')
plt.plot(teste['taxa'], label='Teste', color='orange')
plt.plot(teste.index, previsoes, label='Previsão', color='red', linestyle='--')
plt.title('Previsão ARIMA')
plt.legend()
plt.show()

# Analisar resíduos
residuos = modelo_ajustado.resid
plt.figure(figsize=(10, 6))
plt.plot(residuos)
plt.title('Resíduos do Modelo ARIMA')
plt.show()

# Plot ACF e PACF dos resíduos
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(residuos, ax=plt.gca())
plt.subplot(122)
plot_pacf(residuos, ax=plt.gca())
plt.show()
