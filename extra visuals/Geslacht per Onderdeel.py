import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuratie ---
FILE_PATH = 'dataset.xlsx'
SHEET_NAME = 'Tabel 1'              # De specifieke sheet waar je data staat
HEADER_ROW = 3                      # <--- AANGEPAST: Regelnummer (0-indexed) waar de headers staan. Regel 4 is index 3.
CATEGORY_COLUMN = 'Defensieonderdeel' # Jouw categorie kolom voor de Y-as
FILTER_COLUMN = 'Kenmerk'           # Kolom om te filteren op 'Geslacht'
FILTER_VALUE = 'Geslacht'           # Waarde in FILTER_COLUMN

CATEGORY_FOR_PIVOT = 'Categorie'    # De kolom die 'Man' en 'Vrouw' bevat
PIVOT_VALUES_TO_KEEP = ['Man', 'Vrouw'] # De specifieke waarden die je wilt pivoteren
VALUE_COLUMN_FOR_PIVOT = 'Percentage'   # Dit is de kolom 'Percentage'

# Kolomnamen NA het pivoteren (deze komen overeen met PIVOT_VALUES_TO_KEEP)
LEFT_SIDE_COLUMN_AFTER_PIVOT = 'Man'
RIGHT_SIDE_COLUMN_AFTER_PIVOT = 'Vrouw'

PLOT_TITLE = 'Verdeling van Geslacht (Percentage) per Defensieonderdeel'
X_LABEL = 'Percentage (%)'
Y_LABEL = 'Defensieonderdeel'
LEFT_COLOR = 'skyblue'
RIGHT_COLOR = 'salmon'
LEFT_LABEL = 'Mannen (%)'
RIGHT_LABEL = 'Vrouwen (%)'

# --- Data importeren ---
try:
    # Lees de specifieke sheet en specificeer de header rij
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, header=HEADER_ROW) # <--- BELANGRIJKE AANPASSING HIER
    print(f"Dataset van sheet '{SHEET_NAME}' succesvol geladen (headers van rij {HEADER_ROW + 1}). Eerste 5 rijen (voor filtering):")
    print(df.head())
    print("\nAlle kolomnamen in de dataset:")
    print(df.columns.tolist())
except FileNotFoundError:
    print(f"Fout: Het bestand '{FILE_PATH}' is niet gevonden.")
    print("Controleer of het bestand in dezelfde map staat als dit script, of geef het volledige pad op.")
    exit()
except ValueError:
    print(f"Fout: De sheet '{SHEET_NAME}' is niet gevonden in '{FILE_PATH}'.")
    print("Controleer de naam van de sheet.")
    exit()
except Exception as e:
    print(f"Er is een fout opgetreden bij het lezen van het bestand: {e}")
    exit()

# --- Data filteren ---
required_initial_columns = [CATEGORY_COLUMN, FILTER_COLUMN, CATEGORY_FOR_PIVOT, VALUE_COLUMN_FOR_PIVOT]
if not all(col in df.columns for col in required_initial_columns):
    print(f"\nFout: Een of meer van de benodigde kolommen voor initiële filtering/pivotering zijn niet gevonden in de dataset.")
    print(f"Verwachte kolommen: {required_initial_columns}")
    print(f"Gevonden kolommen: {df.columns.tolist()}")
    print("Controleer de configuratievariabelen (CATEGORY_COLUMN, FILTER_COLUMN, CATEGORY_FOR_PIVOT, VALUE_COLUMN_FOR_PIVOT) en de 'HEADER_ROW'.")
    exit()

# Eerste filter: 'Kenmerk' == 'Geslacht'
df_filtered_kenmerk = df[df[FILTER_COLUMN] == FILTER_VALUE].copy()

if df_filtered_kenmerk.empty:
    print(f"\nWaarschuwing: Na filtering op '{FILTER_COLUMN}' == '{FILTER_VALUE}' is er geen data over.")
    print("Controleer of de filterwaarde en kolomnaam correct zijn, inclusief hoofdlettergevoeligheid.")
    exit()

print(f"\nDataset na filtering op '{FILTER_COLUMN}' == '{FILTER_VALUE}'. Eerste 5 rijen:")
print(df_filtered_kenmerk.head())

# Tweede filter: 'Categorie' is 'Man' of 'Vrouw'
df_filtered_categorie = df_filtered_kenmerk[
    df_filtered_kenmerk[CATEGORY_FOR_PIVOT].isin(PIVOT_VALUES_TO_KEEP)
].copy()

if df_filtered_categorie.empty:
    print(f"\nWaarschuwing: Na filtering op '{CATEGORY_FOR_PIVOT}' in {PIVOT_VALUES_TO_KEEP} is er geen data over.")
    print("Controleer of de filterwaarden en kolomnaam correct zijn, inclusief hoofdlettergevoeligheid.")
    exit()

print(f"\nDataset na filtering op '{CATEGORY_FOR_PIVOT}' in {PIVOT_VALUES_TO_KEEP}. Eerste 5 rijen:")
print(df_filtered_categorie.head())

# Controleer of de Percentage kolom numeriek is
if not pd.api.types.is_numeric_dtype(df_filtered_categorie[VALUE_COLUMN_FOR_PIVOT]):
    print(f"\nFout: De kolom '{VALUE_COLUMN_FOR_PIVOT}' is niet numeriek. Deze moet percentages bevatten.")
    print("Controleer de data in je Excel bestand.")
    exit()


# --- Data pivoteren ---
try:
    df_pivot = df_filtered_categorie.pivot_table(
        index=CATEGORY_COLUMN,
        columns=CATEGORY_FOR_PIVOT,
        values=VALUE_COLUMN_FOR_PIVOT
    ).reset_index()

    df_pivot.columns.name = None

    if LEFT_SIDE_COLUMN_AFTER_PIVOT not in df_pivot.columns or \
       RIGHT_SIDE_COLUMN_AFTER_PIVOT not in df_pivot.columns:
        print("\nFout: Na het pivoteren zijn de kolommen 'Man' of 'Vrouw' niet gevonden.")
        print(f"Verwachte gepivoteerde kolommen: '{LEFT_SIDE_COLUMN_AFTER_PIVOT}', '{RIGHT_SIDE_COLUMN_AFTER_PIVOT}'")
        print(f"Gevonden kolommen na pivoteren: {df_pivot.columns.tolist()}")
        print("Controleer 'PIVOT_VALUES_TO_KEEP' en de data in je Excel bestand.")
        exit()

    # Ensure the pivoted columns are numeric (e.g., if they were mixed types or contained NaNs)
    df_pivot[LEFT_SIDE_COLUMN_AFTER_PIVOT] = pd.to_numeric(df_pivot[LEFT_SIDE_COLUMN_AFTER_PIVOT], errors='coerce').fillna(0)
    df_pivot[RIGHT_SIDE_COLUMN_AFTER_PIVOT] = pd.to_numeric(df_pivot[RIGHT_SIDE_COLUMN_AFTER_PIVOT], errors='coerce').fillna(0)


    # Controleer of er nog steeds geldige numerieke waarden zijn voor plotting
    if df_pivot[LEFT_SIDE_COLUMN_AFTER_PIVOT].sum() == 0 and df_pivot[RIGHT_SIDE_COLUMN_AFTER_PIVOT].sum() == 0:
        print("\nWaarschuwing: Na pivoteren en omzetten naar numeriek, zijn alle waarden voor 'Man' en 'Vrouw' nul. De grafiek zal leeg zijn.")


    print("\nDataset na pivoteren. Eerste 5 rijen:")
    print(df_pivot.head())

except Exception as e:
    print(f"\nFout bij het pivoteren van de data: {e}")
    print("Controleer of de kolomnamen voor pivotering correct zijn en of je data geschikt is voor deze transformatie.")
    exit()


# --- Data voorbereiden voor de butterfly chart ---
categories = df_pivot[CATEGORY_COLUMN].tolist()
left_values = df_pivot[LEFT_SIDE_COLUMN_AFTER_PIVOT].values
right_values = df_pivot[RIGHT_SIDE_COLUMN_AFTER_PIVOT].values

# Maak de waarden voor de linkerkant negatief
left_values_negative = -left_values

# --- Butterfly Chart maken ---
fig, ax = plt.subplots(figsize=(10, 6))

# Teken de staven voor de linkerkant
ax.barh(categories, left_values_negative, color=LEFT_COLOR, label=LEFT_LABEL)

# Teken de staven voor de rechterkant
ax.barh(categories, right_values, color=RIGHT_COLOR, label=RIGHT_LABEL)

# Pas de X-as labels aan om absolute waarden te tonen
max_abs_val = max(abs(left_values).max(), right_values.max())

# Genereer ticks en labels (aangepast voor percentages)
num_ticks = 5 # Aantal ticks aan één kant (exclusief 0)
# Bereken een geschikte stapgrootte die afrondt op een "mooi" getal
if max_abs_val > 0:
    scale = 10**(np.floor(np.log10(max_abs_val / num_ticks)))
    step = np.ceil((max_abs_val / num_ticks) / scale) * scale
    # Zorg dat de stap niet 0 is als de max_abs_val heel klein is
    if step == 0: step = 1
else:
    step = 1 # Als alle waarden 0 zijn, gebruik een stap van 1

x_ticks = np.arange(-max_abs_val - step, max_abs_val + step, step)
x_ticks = x_ticks[np.where(abs(x_ticks) <= max_abs_val * 1.05 + step)]
x_ticks = np.unique(x_ticks)

# Maak de labels: absolute waarden van de ticks, voeg een '%' teken toe
x_labels = [f"{int(abs(tick))}%" for tick in x_ticks]

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)


# Voeg titels en labels toe
ax.set_title(PLOT_TITLE)
ax.set_xlabel(X_LABEL)
ax.set_ylabel(Y_LABEL)

# Voeg een legenda toe
ax.legend()

# Voeg een verticale lijn toe op 0 om de scheiding te benadrukken
ax.axvline(0, color='grey', linestyle='--', linewidth=0.8)

# Optioneel: verwijder de rechter en bovenste 'spines' voor een schonere look
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Zorg ervoor dat de labels niet overlappen
plt.tight_layout()

# Toon de grafiek
plt.show()