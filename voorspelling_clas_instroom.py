import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import os


bestand = 'dataset.xlsx'

if not os.path.exists(bestand):
    print("❌ Bestand niet gevonden!")
    exit()


df = pd.read_excel(bestand, sheet_name='Tabel 4', skiprows=3)


df = df.dropna(subset=['Aantal', 'Defensieonderdeel'])
df['Aantal'] = pd.to_numeric(df['Aantal'], errors='coerce')
df['Jaar'] = pd.to_numeric(df['Jaar'], errors='coerce')


df_clas = df[df['Defensieonderdeel'] == 'CLAS']


instroom_per_jaar = df_clas.groupby('Jaar')['Aantal'].sum().reset_index()


X = instroom_per_jaar[['Jaar']]
y = instroom_per_jaar['Aantal']

model = LinearRegression()
model.fit(X, y)


jaar_2023 = np.array([[2023]])
voorspelling_2023 = model.predict(jaar_2023)[0]
print(f"📈 Voorspelling CLAS-instroom voor 2023: {voorspelling_2023:.0f} personen")

instroom_per_jaar['Voorspeld'] = model.predict(X)
instroom_met_2023 = instroom_per_jaar.copy()
instroom_met_2023 = pd.concat([
    instroom_met_2023,
    pd.DataFrame({'Jaar': [2023], 'Aantal': [None], 'Voorspeld': [voorspelling_2023]})
], ignore_index=True)


plt.figure(figsize=(10, 6))
plt.plot(instroom_met_2023['Jaar'], instroom_met_2023['Voorspeld'], label='Voorspeld', color='orange', linestyle='--')
plt.scatter(instroom_per_jaar['Jaar'], instroom_per_jaar['Aantal'], label='Werkelijke instroom', color='blue')
plt.title('Voorspelling Instroom CLAS (2018–2023)')
plt.xlabel('Jaar')
plt.ylabel('Instroom')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
