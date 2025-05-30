import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import os

# Bestandspad
bestand = 'dataset.xlsx'

# Controleren of bestand bestaat
if not os.path.exists(bestand):
    print("❌ Bestand niet gevonden!")
    exit()

# Inlezen vanaf rij 4
df = pd.read_excel(bestand, sheet_name='Tabel 4', skiprows=3)

# Converteren en opschonen
df['Aantal'] = pd.to_numeric(df['Aantal'], errors='coerce')
df['Jaar'] = pd.to_numeric(df['Jaar'], errors='coerce')
df = df.dropna(subset=['Aantal', 'Jaar', 'Defensieonderdeel', 'Type personeel'])

# Filteren
df_filtered = df[
    (df['Defensieonderdeel'] == 'CLAS') &
    (df['Type personeel'] == 'Totaal')
]

# X en y
X = df_filtered[['Jaar']]
y = df_filtered['Aantal']

# Jaar range voor grafiek
jaar_range = np.arange(X['Jaar'].min(), 2024).reshape(-1, 1)

# === 1. Lineaire regressie ===
lin_model = LinearRegression()
lin_model.fit(X, y)
voorsp_2023_lin = lin_model.predict([[2023]])
lijn_lin = lin_model.predict(jaar_range)

# === 2. Random Forest ===
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
voorsp_2023_rf = rf_model.predict([[2023]])
lijn_rf = rf_model.predict(jaar_range)

# === 3. MLP Regressor (Neuraal Netwerk) ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
jaar_range_scaled = scaler.transform(jaar_range)
jaar_2023_scaled = scaler.transform([[2023]])

mlp_model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
mlp_model.fit(X_scaled, y)
voorsp_2023_mlp = mlp_model.predict(jaar_2023_scaled)
lijn_mlp = mlp_model.predict(jaar_range_scaled)

# === Resultaten printen ===
print(f"\n📊 Voorspellingen voor 2023:")
print(f"🔹 Lineaire regressie     : {voorsp_2023_lin[0]:.2f}")
print(f"🔹 Random Forest          : {voorsp_2023_rf[0]:.2f}")
print(f"🔹 Neuraal netwerk (MLP)  : {voorsp_2023_mlp[0]:.2f}")

# === Visualisatie ===
plt.figure(figsize=(12, 7))
plt.scatter(X, y, color='black', label='Historische data')

# Lijnen
plt.plot(jaar_range, lijn_lin, color='blue', linestyle='--', label='Lineaire regressie')
plt.plot(jaar_range, lijn_rf, color='green', linestyle='-', label='Random Forest')
plt.plot(jaar_range, lijn_mlp, color='purple', linestyle=':', label='Neuraal netwerk (MLP)')

# Punten 2023
plt.scatter(2023, voorsp_2023_lin, color='blue', edgecolors='k', s=100, label='Voorsp. 2023 - Lineair')
plt.scatter(2023, voorsp_2023_rf, color='green', edgecolors='k', s=100, label='Voorsp. 2023 - RF')
plt.scatter(2023, voorsp_2023_mlp, color='purple', edgecolors='k', s=100, label='Voorsp. 2023 - MLP')

# Layout
plt.xlabel('Jaar')
plt.ylabel('Aantal')
plt.title('Vergelijking regressiemodellen voor instroom CLAS (Totaal personeel)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
