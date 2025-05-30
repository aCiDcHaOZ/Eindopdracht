import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import os

# Bestandspad
bestand = 'dataset.xlsx'

# Check of bestand bestaat
if not os.path.exists(bestand):
    print("❌ Bestand niet gevonden!")
    exit()

# Inlezen vanaf rij 4
df = pd.read_excel(bestand, sheet_name='Tabel 4', skiprows=3)

# Kolommen converteren
df['Aantal'] = pd.to_numeric(df['Aantal'], errors='coerce')
df['Jaar'] = pd.to_numeric(df['Jaar'], errors='coerce')

# Verwijder onvolledige rijen
df = df.dropna(subset=['Aantal', 'Jaar', 'Defensieonderdeel', 'Type personeel'])

# Filters toepassen
df_filtered = df[
    (df['Defensieonderdeel'] == 'CLAS') &
    (df['Type personeel'] == 'Totaal')
]

# X en y definiëren (geen groepering!)
X = df_filtered[['Jaar']]
y = df_filtered['Aantal']

# Model trainen
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Voorspelling voor 2023
jaar_2023 = np.array([[2023]])
voorspelling_2023 = model.predict(jaar_2023)
print(f"📈 Voorspelde instroom in 2023 (Random Forest): {voorspelling_2023[0]:.0f} personen")

# Visualisatie
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Historische data')

# Regressielijn genereren over bereik van jaren
jaar_min = int(X['Jaar'].min())
jaar_max = 2023  # tot en met 2023 inclusief voorspelling
jaar_range = np.arange(jaar_min, jaar_max + 1).reshape(-1, 1)
voorspellingen_range = model.predict(jaar_range)
plt.plot(jaar_range, voorspellingen_range, color='green', label='Modelvoorspelling (RF)')

# Voorspelling voor 2023 markeren
plt.scatter(2023, voorspelling_2023, color='red', s=100, label='Voorspelling 2023')

# Layout
plt.xlabel('Jaar')
plt.ylabel('Aantal')
plt.title('Random Forest Regressie – Instroom CLAS (Totaal personeel)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()