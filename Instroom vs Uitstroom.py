import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- DataFrames voor instroom en uitstroom (voorbeelddata verwijderd) ---
# Plaats hier je eigen instroom_df DataFrame.
# Zorg ervoor dat het kolommen 'Jaar', 'Kenmerk' en 'Aantal' bevat.
# Voorbeeld:
# instroom_data = {
#     'Jaar': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
#     'Kenmerk': ['Totaal', 'Totaal', 'Totaal', 'Totaal', 'Totaal', 'Totaal', 'Totaal'],
#     'Aantal': [2500, 2750, 2450, 2950, 3100, 3300, 3500]
# }
# instroom_df = pd.DataFrame(instroom_data)

# Als je een complexere structuur hebt zoals voorheen, filter dan 'Totaal':
# instroom_df = pd.DataFrame({
#     'Jaar': [2018, 2018, 2019, 2019],
#     'Kenmerk': ['Nieuwe_Studenten', 'Totaal', 'Nieuwe_Studenten', 'Totaal'],
#     'Aantal': [1500, 2500, 1650, 2750]
# })
# instroom_totaal_df = instroom_df[instroom_df['Kenmerk'] == 'Totaal'].copy()

# Voor dit voorbeeld gebruiken we de gefilterde data direct om de code werkend te houden.
# Vervang dit met je eigen data laadlogica.
instroom_totaal_df = pd.DataFrame({
    'Jaar': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Aantal': [2500, 2750, 2450, 2950, 3100, 3300, 3500]
})


print("Gefilterd DataFrame 'instroom_totaal_df' (Kenmerk == 'Totaal'):")
print(instroom_totaal_df)
print("-" * 30)

# Plaats hier je eigen uitstroom_df DataFrame.
# Zorg ervoor dat het kolommen 'Jaar', 'Kenmerk' en 'Aantal' bevat.
# Voorbeeld:
# uitstroom_data = {
#     'Jaar': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
#     'Kenmerk': ['Totaal', 'Totaal', 'Totaal', 'Totaal', 'Totaal', 'Totaal', 'Totaal'],
#     'Aantal': [1500, 1750, 1650, 2000, 2250, 2450, 2650]
# }
# uitstroom_df = pd.DataFrame(uitstroom_data)

# Voor dit voorbeeld gebruiken we de gefilterde data direct om de code werkend te houden.
# Vervang dit met je eigen data laadlogica.
uitstroom_totaal_df = pd.DataFrame({
    'Jaar': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Aantal': [1500, 1750, 1650, 2000, 2250, 2450, 2650]
})


print("Gefilterd DataFrame 'uitstroom_totaal_df' (Kenmerk == 'Totaal'):")
print(uitstroom_totaal_df)
print("-" * 30)


plt.figure(figsize=(12, 7)) # Stel de grootte van de grafiek in
ax = plt.gca() # Haal de huidige assen op

# Plot de instroom lijn (nu groen)
instroom_totaal_df.plot(x='Jaar', y='Aantal', kind='line', ax=ax, marker='o', color='green', linewidth=2, label='Instroom Totaal')

# Plot de uitstroom lijn op dezelfde assen (nu rood en solide)
uitstroom_totaal_df.plot(x='Jaar', y='Aantal', kind='line', ax=ax, marker='x', color='red', linewidth=2, linestyle='-', label='Uitstroom Totaal')

# Bereken het verschil tussen instroom en uitstroom
# Zorg ervoor dat de DataFrames op 'Jaar' zijn gesorteerd voor correcte aftrekking
instroom_totaal_df = instroom_totaal_df.sort_values(by='Jaar').reset_index(drop=True)
uitstroom_totaal_df = uitstroom_totaal_df.sort_values(by='Jaar').reset_index(drop=True)

# Voeg de 'Aantal' kolommen samen en bereken het verschil
# Een merge is robuuster als jaren niet perfect overeenkomen, maar voor dit voorbeeld is directe aftrekking oké
# Eerst zorgen we dat de 'Jaar' kolommen overeenkomen, dan de 'Aantal' kolommen aftrekken
gecombineerd_df = pd.merge(instroom_totaal_df, uitstroom_totaal_df, on='Jaar', suffixes=('_Instroom', '_Uitstroom'))
gecombineerd_df['Verschil'] = gecombineerd_df['Aantal_Instroom'] - gecombineerd_df['Aantal_Uitstroom']

# Plot de verschil-lijn (nu paars en stippellijn)
gecombineerd_df.plot(x='Jaar', y='Verschil', kind='line', ax=ax, marker='s', color='purple', linewidth=2, linestyle='--', label='Verschil (Instroom - Uitstroom)')

plt.title('Totaal Aantal Instroom, Uitstroom en Verschil per Jaar') # Titel van de grafiek
plt.xlabel('Jaar') # Label voor de X-as
plt.ylabel('Totaal Aantal') # Label voor de Y-as
# Combineer de jaren van alle dataframes om alle jaren op de X-as te tonen
all_years = sorted(list(set(instroom_totaal_df['Jaar'].tolist() + uitstroom_totaal_df['Jaar'].tolist())))
plt.xticks(all_years) # Zorg ervoor dat alle jaren op de X-as worden weergegeven
plt.grid(True, linestyle='--', alpha=0.7) # Voeg een raster toe
plt.legend(title='Categorie') # Voeg een legenda toe
plt.tight_layout() # Pas de lay-out aan
plt.show()