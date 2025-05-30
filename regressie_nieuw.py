import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# Bestandspad
bestand = 'dataset.xlsx'

# Check of bestand bestaat
if not os.path.exists(bestand):
    print("❌ Bestand niet gevonden!")
    exit()

# Inlezen van Excel (koppen op rij 4 = skiprows=3)
df = pd.read_excel(bestand, sheet_name='Tabel 4', skiprows=3)

# Kolommen converteren
df['Aantal'] = pd.to_numeric(df['Aantal'], errors='coerce')
df['Jaar'] = pd.to_numeric(df['Jaar'], errors='coerce')

# Verwijder rijen met NaN
df = df.dropna(subset=['Aantal', 'Jaar', 'Defensieonderdeel', 'Type personeel'])

# Filters toepassen
df_filtered = df[
    (df['Defensieonderdeel'] == 'CLAS') &
    (df['Type personeel'] == 'Totaal')
]

# X en y toewijzen
X = df_filtered[['Jaar']]
y = df_filtered['Aantal']

# Regressiemodel trainen
model = LinearRegression()
model.fit(X, y)

# Voorspelling voor 2023
jaar_2023 = np.array([[2023]])
voorspelling_2023 = model.predict(jaar_2023)
print(f"📈 Voorspelde instroom in 2023: {voorspelling_2023[0]:.0f} personen")

# Plotten
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data CLAS - Totaal personeel')
plt.plot(X, model.predict(X), color='green', label='Regressielijn')
plt.scatter(2023, voorspelling_2023, color='red', s=100, label='Voorspelling 2023')

plt.xlabel('Jaar')
plt.ylabel('Aantal')
plt.title('Regressieanalyse CLAS instroom (niet gegroepeerd)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
