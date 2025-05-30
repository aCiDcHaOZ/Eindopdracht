import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Dataset inladen en voorbereiden
file_path = 'dataset.xlsx'
sheet_name = 'Tabel 4'

# De dataset inladen, rekening houdend met de headers op rij 4 (index 3)
try:
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=3)
except FileNotFoundError:
    print(f"Fout: Het bestand '{file_path}' is niet gevonden. Zorg ervoor dat het bestand in dezelfde directory staat als je Python-script.")
    exit()

# Kolomnamen controleren en indien nodig aanpassen voor spaties en speciale tekens
df.columns = df.columns.str.strip()

# 2. Filteren op 'Defensieonderdeel' = 'CLAS'
df_filtered = df[df['Defensieonderdeel'] == 'CLAS'].copy()

# 3. NaN-regels verwijderen in de Y-variabele 'Aantal'
df_filtered.dropna(subset=['Aantal'], inplace=True)

# Zorg ervoor dat 'Aantal' een numeriek type is
df_filtered['Aantal'] = pd.to_numeric(df_filtered['Aantal'], errors='coerce')
df_filtered.dropna(subset=['Aantal'], inplace=True) # Verwijder nu eventuele NaN's die door de conversie zijn ontstaan

# 4. X- en Y-variabelen definiëren
X = df_filtered[['Jaar']]
y = df_filtered['Aantal']

# 5. Train/Test Split (optioneel, maar goed voor evaluatie)
# Voor dit specifieke geval van voorspelling van één jaar, en als je niet wilt evalueren op historische data,
# kun je deze stap overslaan en direct trainen op de volledige dataset.
# Echter, voor een robuuster model is een split aan te raden als je voldoende data hebt.
# Hier splitsen we om te laten zien hoe de verwarringsmatrix werkt (hoewel deze meer voor classificatie is).
# Voor regressie is een verwarringsmatrix niet direct van toepassing. We zullen in plaats daarvan een scatterplot maken
# en regressie-evaluatiemetrieken gebruiken.
# We zullen een "pseudo-verwarringsmatrix" maken door de voorspelde waarden te categoriseren om een visuele weergave te krijgen.
# Echter, voor pure regressie is een verwarringsmatrix niet het meest geschikte evaluatietool.
# Ik zal een scatterplot maken van voorspelde vs. werkelijke waarden en de MAE en R2-score berekenen.

# Voor het maken van een "verwarringsmatrix" voor regressie is het nodig om de continue waarden te categoriseren.
# Dit is echter geen standaardpraktijk. Een scatterplot en metrische zoals MAE en R2 zijn gebruikelijker.
# Als je per se een verwarringsmatrix wilt, moet je de 'Aantal' variabele binned (gecategoriseerd) maken.
# Ik zal een voorbeeld geven van hoe je dit zou kunnen doen voor demonstratiedoeleinden, maar waarschuw dat dit
# voor regressie ongebruikelijk is.

# Om toch iets van een "verwarringsmatrix" te tonen, zullen we de 'Aantal' waarden binned maken.
# Dit is meer ter illustratie dan een standaard evaluatie voor regressie.
# Kies hierbij passende bins op basis van de distributie van je 'Aantal' variabele.
bins = [0, 50, 100, 150, 200, np.inf] # Voorbeeld bins
labels = ['0-50', '51-100', '101-150', '151-200', '>200']
y_binned = pd.cut(y, bins=bins, labels=labels, right=False)

# Splits de data voor training en testen (nodig voor evaluatie)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Random Forest Regressor model trainen
model = RandomForestRegressor(n_estimators=100, random_state=42) # n_estimators: aantal bomen in het bos
model.fit(X_train, y_train)

# 7. Voorspellingen maken voor het jaar 2023
jaar_2023_df = pd.DataFrame({'Jaar': [2023]})
voorspelling_2023 = model.predict(jaar_2023_df)

print(f"\nVoorspelling voor 'Aantal' in het jaar 2023: {voorspelling_2023[0]:.2f}\n")

# 8. Model evaluatie (voor de trainingsdata)
# Voorspellingen op de testset
y_pred = model.predict(X_test)

# Bereken Mean Absolute Error (MAE) en R2 Score
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE) op de testset: {mae:.2f}")
print(f"R-squared (R2) score op de testset: {r2:.2f}\n")

# Visualisatie van voorspellingen vs. werkelijke waarden (Scatterplot)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Ideale lijn
plt.xlabel('Werkelijk Aantal')
plt.ylabel('Voorspeld Aantal')
plt.title('Werkelijk vs. Voorspeld Aantal (Testset)')
plt.grid(True)
plt.show()

# "Verwarringsmatrix" voor regressie (met categorisatie)
# Zoals eerder vermeld, is een verwarringsmatrix meer voor classificatie.
# Voor regressie is dit een ongebruikelijke visualisatie, maar hier ter demonstratie.

# Categorieën maken voor de werkelijke en voorspelde waarden op de testset
y_test_binned = pd.cut(y_test, bins=bins, labels=labels, right=False)
y_pred_binned = pd.cut(y_pred, bins=bins, labels=labels, right=False)

# Maak de "verwarringsmatrix"
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_binned, y_pred_binned, labels=labels) # Zorg dat labels consistent zijn

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Voorspelde Categorie')
plt.ylabel('Werkelijke Categorie')
plt.title('Pseudo Verwarringsmatrix voor Aantal (gecategoriseerd)')
plt.show()

print("\nOpmerking over de verwarringsmatrix: Voor regressie is een verwarringsmatrix ongebruikelijk. Het vereist dat de continue 'Aantal' variabele wordt gecategoriseerd. De scatterplot van werkelijke vs. voorspelde waarden en metrische zoals MAE en R2 zijn veel gangbaardere evaluatietools voor regressieproblemen.")