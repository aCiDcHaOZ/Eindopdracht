import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1. Gegevens laden en filteren ---
try:
    df = pd.read_excel('dataset.xlsx', sheet_name='Tabel 1', header=3)
    print("Gegevens succesvol geladen met kolomtitels van rij 4.")
    print(f"Initiële rijen in DataFrame: {len(df)}")
    print(df.head())

    # Debugging tip: print hier de gedetecteerde kolomnamen om zeker te zijn van de spelling
    print("\nGedetecteerde kolomnamen na inlezen (ter controle):")
    print(df.columns.tolist())
    print("-" * 50)

    # Gegevens filteren op specifieke kolommen
    filter_columns = {
        'Kenmerk': 'Totaal',
        'Type personeel': 'Totaal',
        'Defensieonderdeel': 'Totaal'
    }

    for col, value in filter_columns.items():
        if col in df.columns:
            df = df[df[col] == value]
            print(f"Gefilterd op '{col}' == '{value}'. Rijen over: {len(df)}")
        else:
            print(f"Waarschuwing: Kolom '{col}' niet gevonden in de dataset. Filter niet toegepast voor deze kolom.")

    if len(df) == 0:
        print("\nFout: Na het filteren zijn er geen rijen meer over. Controleer de filterwaarden en kolomnamen.")
        exit()

    # Zorg ervoor dat 'Jaar' een numerieke kolom is en sorteer de DataFrame
    if 'Jaar' in df.columns:
        df['Jaar'] = pd.to_numeric(df['Jaar'], errors='coerce')
        df.sort_values(by='Jaar', inplace=True)
        print("\nDataFrame gesorteerd op 'Jaar'.")
    else:
        print("Waarschuwing: Kolom 'Jaar' niet gevonden. Kan niet sorteren op jaar.")

except FileNotFoundError:
    print("Fout: 'dataset.xlsx' niet gevonden. Zorg ervoor dat het bestand in dezelfde map staat als je script.")
    exit()

# --- 2. Gegevens voorbereiden en lagged features creëren ---
target_column = 'Totaal'
direct_feature_columns = []
num_lags = 1 # Overweeg dit te verhogen als je meer data hebt en meer context wilt

if target_column not in df.columns:
    print(f"Fout: De target_column '{target_column}' is niet gevonden in de dataset. Controleer de kolomnaam.")
    print(f"Beschikbare kolommen: {df.columns.tolist()}")
    exit()

# Maak lagged features voor de target ('Totaal')
for i in range(1, num_lags + 1):
    df[f'{target_column}_lag_{i}'] = df[target_column].shift(i)

# Verwijder de eerste rijen die NaN's bevatten door de shift operatie
# Deze dropna is cruciaal voor de lagged features
df.dropna(inplace=True)
print(f"Aantal rijen na droppen van NaN's door lags: {len(df)}")


# Maak de uiteindelijke feature- en target-sets
features_with_lags = [f'{target_column}_lag_{i}' for i in range(1, num_lags + 1)]

if not features_with_lags:
    print("Fout: Geen lagged features gecreëerd. Controleer 'num_lags' en de data.")
    exit()

features = df[features_with_lags]
target = df[target_column]

# Zorg ervoor dat alle gegevens numeriek zijn
features = features.apply(pd.to_numeric, errors='coerce')
target = pd.to_numeric(target, errors='coerce')

# Vervang oneindige waarden of NaN's die zijn ontstaan door delingen door nul in features en target
features.replace([np.inf, -np.inf], np.nan, inplace=True)
target.replace([np.inf, -np.inf], np.nan, inplace=True)

# Vul eventuele resterende NaN's op met het gemiddelde
features.fillna(features.mean(), inplace=True)
target.fillna(target.mean(), inplace=True)

# --- Extra controle op NaN's in features of target vóór split ---
if features.isnull().sum().sum() > 0:
    print(f"\nWAARSCHUWING: NaN's gevonden in de features data VOOR de split. Aantal: {features.isnull().sum().sum()}")
if target.isnull().sum() > 0:
    print(f"WAARSCHUWING: NaN's gevonden in de target data VOOR de split. Aantal: {target.isnull().sum()}")

print(f"\nFeatures geselecteerd (alleen lags van '{target_column}'): {features.columns.tolist()}")
print(f"Target geselecteerd: {target_column}")
print(f"Aantal rijen voor train/test split: {len(features)}")


# --- 3. Data splitsen in training- en testsets ---
# Controleer of er genoeg data is om te splitsen
# Minimaal 2 rijen voor training en 1 rij voor test (of vice versa)
if len(features) < 2:
    print("\nFout: Te weinig data na preprocessing om te splitsen in training- en testsets. Er zijn minder dan 2 rijen data over. Controleer de dataset en filters.")
    exit()

# Controleer of test_size niet te groot is voor de beschikbare data
if len(features) * 0.2 < 1: # Zorg dat er minimaal 1 test sample is
    print("\nWaarschuwing: Testset wordt te klein (<1 sample). Pas 'test_size' aan of controleer de data.")
    # Je zou hier test_size = 1/len(features) kunnen zetten om toch 1 sample te krijgen
    # Of gewoon exit() als de data te beperkt is
    test_size_actual = 1 / len(features) if len(features) >= 1 else 0 # Minimum 1 sample in testset
    print(f"Test size aangepast naar: {test_size_actual:.2f}")
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size_actual, random_state=42)
else:
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


print(f"\nAantal trainingssamples na split: {len(X_train)}")
print(f"Aantal testsamples na split: {len(X_test)}")

if len(X_train) == 0 or len(X_test) == 0:
    print("\nFout: Trainings- of testset is leeg na het splitsen. Dit wijst op te weinig data.")
    exit()

# --- 4. Gegevens schalen ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. Neuraal netwerk bouwen ---
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1) # Outputlaag voor regressie
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("\nModeloverzicht:")
model.summary()

# --- 6. Model trainen ---
# Gebruik X_train_scaled.shape[0] om te zien hoeveel samples er zijn
if X_train_scaled.shape[0] == 0:
    print("\nFout: Geen trainingsdata voor het model. Controleer preprocessing en splitsing.")
    exit()
# Zet verbose op 0 of 1; 2 is te veel voor een korte training
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)

print("\nTraining voltooid.")

# --- 7. Model evalueren ---
if X_test_scaled.shape[0] == 0:
    print("\nFout: Geen testdata om voorspellingen op te doen. Controleer preprocessing en splitsing.")
    exit()

y_pred = model.predict(X_test_scaled)

# Converteer y_pred naar een 1D array voor consistente vergelijking
y_pred = y_pred.flatten()

# --- Finale controle op NaN's vlak voor metrics ---
# Zorg ervoor dat y_test en y_pred dezelfde lengte hebben en geen NaN's bevatten
# Dit is de meest kritieke stap om de ValueError te voorkomen.
# We zoeken naar de indices waar beide arrays GEEN NaN bevatten
valid_indices = ~np.isnan(y_test) & ~np.isnan(y_pred)

print(f"\nAantal geldige indices voor evaluatie: {valid_indices.sum()}") # Nieuwe debug print

if valid_indices.sum() == 0:
    print("\nFout: Geen geldige (non-NaN) data-punten over voor evaluatie na voorspelling. Controleer de data.")
    # Extra debug info:
    print(f"NaN's in y_test: {np.isnan(y_test).sum()} van {len(y_test)}")
    print(f"NaN's in y_pred: {np.isnan(y_pred).sum()} van {len(y_pred)}")
    print(f"Lengte y_test: {len(y_test)}, Lengte y_pred: {len(y_pred)}")
    exit()

y_test_clean = y_test[valid_indices]
y_pred_clean = y_pred[valid_indices]

mse = mean_squared_error(y_test_clean, y_pred_clean)
r2 = r2_score(y_test_clean, y_pred_clean)

print(f"\nMean Squared Error (MSE) op testdata: {mse:.2f}")
print(f"R-kwadraat (R2) op testdata: {r2:.2f}")

# Plot de training/validatie loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot voorspelde vs. werkelijke waarden
plt.figure(figsize=(10, 6))
plt.scatter(y_test_clean, y_pred_clean, alpha=0.7) # Gebruik de schone data
plt.plot([y_test_clean.min(), y_test_clean.max()], [y_test_clean.min(), y_test_clean.max()], 'r--', lw=2)
plt.title('Werkelijke vs. Voorspelde Totaal')
plt.xlabel('Werkelijke Totaal')
plt.ylabel('Voorspelde Totaal')
plt.grid(True)
plt.show()