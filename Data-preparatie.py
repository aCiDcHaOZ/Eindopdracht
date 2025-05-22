
import pandas as pd
import matplotlib.pyplot as plt
import os



bestand = 'dataset.xlsx'

if not os.path.exists(bestand):
    print("❌ Bestand niet gevonden!")
    exit()

df = pd.read_excel(bestand, sheet_name='Tabel 4', skiprows=3)

# Regels met een lege entry bij kolom 'Aantal' of 'Defensieonderdeel' verwijderen
df = df.dropna(subset=['Aantal', 'Defensieonderdeel'])

# Waardes in kolom 'Aantal' nummeriek maken, waardes met error vervangen voor NaN
df['Aantal'] = pd.to_numeric(df['Aantal'], errors='coerce')

totaal_instroom = df.groupby('Defensieonderdeel')['Aantal'].sum().sort_values(ascending=False)

print("📊 Top krijgsmachtonderdelen met hoogste instroom (2018–2022):")
print(totaal_instroom.head(10))

plt.figure(figsize=(10, 6))
totaal_instroom.plot(kind='bar', color='steelblue')
plt.title('Instroom per Krijgsmachtonderdeel (2018–2022)')
plt.xlabel('Krijgsmachtonderdeel')
plt.ylabel('Totaal Instroom')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
