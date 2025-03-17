# -*- coding: utf-8 -*-
"""
Versão aprimorada do modelo de previsão da TR
"""

#%% Importação das bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

#%% Carregamento dos dados
tr_data = pd.read_parquet('datas/base_dados_tr.parquet')

#%% Separação entre features (X) e variável target (y)
X = tr_data[['selic', 'ipca', 'dolar']]
y = tr_data['tr']  # Mantido como Série para evitar normalização

#%% Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#%% Normalização das features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

#%% Modelo 1: Regressão Linear
reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train)

y_pred = reg_model.predict(X_test_scaled)

# Avaliação do modelo
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f'\nDesempenho do modelo de Regressão Linear:')
print(f'R² Score: {r2:.3f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')

#%% Validação cruzada
cv_scores = cross_val_score(reg_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f'R² médio com validação cruzada: {np.mean(cv_scores):.3f}')

#%% Modelo 2: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_rf_pred = rf_model.predict(X_test_scaled)

# Avaliação do modelo Random Forest
r2_rf = r2_score(y_test, y_rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_rf_pred))
mae_rf = mean_absolute_error(y_test, y_rf_pred)

print(f'\nDesempenho do modelo Random Forest:')
print(f'R² Score: {r2_rf:.3f}')
print(f'RMSE: {rmse_rf:.4f}')
print(f'MAE: {mae_rf:.4f}')

#%% Importância das Variáveis - Random Forest
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color='green')
plt.xlabel('Importância')
plt.title('Importância das Features - Random Forest')
plt.show()
