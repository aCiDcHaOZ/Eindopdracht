import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Bestandspad
bestand = 'dataset.xlsx'

# Controleren of bestand bestaat
if not os.path.exists(bestand):
    print("❌ Bestand niet gevonden!")
    exit()

# Inlezen van de data
df = pd.read_excel(bestand, sheet_name='Tabel 4', skiprows=3)

# Kolommen converteren
df['Aantal'] = pd.to_numeric(df['Aantal'], errors='coerce')
df['Jaar'] = pd.to_numeric(df['Jaar'], errors='coerce')

# Filter
df = df.dropna(subset=['Aantal', 'Jaar'])
df_clas = df[df['Defensieonderdeel'] == 'CLAS']

# Features en target
X = df_clas[['Jaar']]
y = df_clas['Aantal']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluatie
y_pred = model.predict(X_test)
print("📊 Evaluatie op testdata:")
print(f"  MAE : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"  MSE : {mean_squared_error(y_test, y_pred):.2f}")
print(f"  R²  : {r2_score(y_test, y_pred):.2f}")

# Voorspelling voor 2023
jaar_2023 = np.array([[2023]])
voorspelling_2023 = model.predict(jaar_2023)
print(f"\n📈 Voorspelling voor 2023: {voorspelling_2023[0]:.0f} personen")

# Visualisatie
plt.figure(figsize=(10, 6))

# Scatterplot van historische data
plt.scatter(X, y, color='blue', label='Historische data')

# Regressielijn over jaren
jaar_range = np.arange(X['Jaar'].min(), 2024).reshape(-1, 1)
voorspellingen_range = model.predict(jaar_range)
plt.plot(jaar_range, voorspellingen_range, color='green', label='Regressielijn (RF)')

# Voorspelling voor 2023 als punt
plt.scatter(2023, voorspelling_2023, color='red', s=100, label='Voorspelling 2023')

# Layout
plt.xlabel('Jaar')
plt.ylabel('Aantal')
plt.title('Voorspelling instroom CLAS met regressielijn')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
