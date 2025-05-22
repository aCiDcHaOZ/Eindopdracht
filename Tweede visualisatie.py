import pandas as pd
import matplotlib.pyplot as plt
import os



bestand = 'dataset.xlsx'

if not os.path.exists(bestand):
    print("❌ Bestand niet gevonden!")
    exit()


df = pd.read_excel(bestand, sheet_name='Tabel 4', skiprows=3)


df = df.dropna(subset=['Aantal', 'Defensieonderdeel'])
df['Aantal'] = pd.to_numeric(df['Aantal'], errors='coerce')
df['Jaar'] = pd.to_numeric(df['Jaar'], errors='coerce')

df_filtered = df[df['Defensieonderdeel'].notna() & (df['Defensieonderdeel'] != 'Totaal')]


pivot = df_filtered.pivot_table(index='Jaar', columns='Defensieonderdeel', values='Aantal', aggfunc='sum')

print("🔍 Pivot-table preview:")
print(pivot.head())


pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Instroom per Jaar per Krijgsmachtonderdeel (2018–2022)')
plt.xlabel('Jaar')
plt.ylabel('Instroom')
plt.xticks(rotation=0)
plt.legend(title='Krijgsmachtonderdeel', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
