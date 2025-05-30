import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuratie ---
FILE_PATH = 'dataset.xlsx'
SHEET_NAME = 'Tabel 4'              # We gebruiken 'Tabel 4'
HEADER_ROW = 3                      # Regelnummer (0-indexed) waar de headers staan. (Regel 4 is index 3)
                                    # Controleer of dit nog steeds geldt voor 'Tabel 4'!

# --- Kolomnamen en filterwaarden ---
# Zorg dat deze namen exact overeenkomen met de kolomnamen in je Excel-bestand!
CATEGORY_COLUMN = 'Leeftijd'          # <--- AANGEPAST: De kolom voor de Y-as is nu 'Leeftijd'
VALUE_COLUMN = 'Percentage'           # De kolom met de numerieke waarden

# Filters die toegepast moeten worden
# Let op: 'Defensieonderdeel' filter op 'Totaal' is verwijderd.
FILTERS = {
    'Type personeel': 'Totaal',
    'Jaar': 2022
}

# Kolom voor pivotering (om 'Man' en 'Vrouw' te scheiden)
GENDER_COLUMN_FOR_PIVOT = 'Geslacht' # <--- AANGEPAST: De kolom die 'Man' en 'Vrouw' bevat

# Kolomnamen NA het pivoteren
LEFT_SIDE_COLUMN_AFTER_PIVOT = 'Man'
RIGHT_SIDE_COLUMN_AFTER_PIVOT = 'Vrouw'

# Titels en labels van de grafiek
PLOT_TITLE = 'Percentage Mannen en Vrouwen per Leeftijdsgroep (Totaal Personeel, 2022)' # Pas titel aan
X_LABEL = 'Percentage (%)'
Y_LABEL = 'Leeftijdsgroep'
LEFT_COLOR = 'skyblue'
RIGHT_COLOR = 'salmon'
LEFT_LABEL = 'Mannen (%)'
RIGHT_LABEL = 'Vrouwen (%)'

# --- Data importeren ---
try:
    df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME, header=HEADER_ROW)
    print(f"Dataset van sheet '{SHEET_NAME}' succesvol geladen (headers van rij {HEADER_ROW + 1}). Eerste 5 rijen:")
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
# Controleer of alle benodigde filterkolommen en de waardekolom bestaan
all_required_cols = list(FILTERS.keys()) + [CATEGORY_COLUMN, VALUE_COLUMN, GENDER_COLUMN_FOR_PIVOT]
if not all(col in df.columns for col in all_required_cols):
    missing_cols = [col for col in all_required_cols if col not in df.columns]
    print(f"\nFout: Een of meer van de benodigde kolommen zijn niet gevonden in de dataset.")
    print(f"Ontbrekende kolommen: {missing_cols}")
    print(f"Gevonden kolommen: {df.columns.tolist()}")
    print("Controleer de configuratievariabelen.")
    exit()

# Pas alle filters toe, behalve de 'Geslacht' filter (die gebruiken we voor pivotering)
df_filtered = df.copy()
for col, val in FILTERS.items():
    df_filtered = df_filtered[df_filtered[col] == val].copy()

# Filter nu alleen op de 'Man' en 'Vrouw' categorieën in de Geslacht kolom
df_filtered = df_filtered[df_filtered[GENDER_COLUMN_FOR_PIVOT].isin(['Man', 'Vrouw'])].copy()

if df_filtered.empty:
    print(f"\nWaarschuwing: Na alle filtering is er geen data over.")
    print(f"Controleer de filterwaarden: {FILTERS} en de waarden voor '{GENDER_COLUMN_FOR_PIVOT}' ('Man', 'Vrouw').")
    print("De grafiek kan niet worden gemaakt.")
    exit()

print(f"\nDataset na filtering. Resultaat (eerste 5 rijen):")
print(df_filtered.head())

# Controleer of de waardekolom numeriek is
if not pd.api.types.is_numeric_dtype(df_filtered[VALUE_COLUMN]):
    print(f"\nFout: De kolom '{VALUE_COLUMN}' is niet numeriek. Deze moet percentages bevatten.")
    print("Controleer de data in je Excel bestand of de naam van de kolom.")
    exit()

# --- Data pivoteren ---
# Pivot de data om 'Man' en 'Vrouw' als aparte kolommen te krijgen per leeftijdscategorie
try:
    df_pivot = df_filtered.pivot_table(
        index=CATEGORY_COLUMN,
        columns=GENDER_COLUMN_FOR_PIVOT,
        values=VALUE_COLUMN
    ).reset_index()

    df_pivot.columns.name = None # Verwijder de naam van de kolom-index

    if LEFT_SIDE_COLUMN_AFTER_PIVOT not in df_pivot.columns or \
       RIGHT_SIDE_COLUMN_AFTER_PIVOT not in df_pivot.columns:
        print("\nFout: Na het pivoteren zijn de kolommen 'Man' of 'Vrouw' niet gevonden.")
        print(f"Verwachte gepivoteerde kolommen: '{LEFT_SIDE_COLUMN_AFTER_PIVOT}', '{RIGHT_SIDE_COLUMN_AFTER_PIVOT}'")
        print(f"Gevonden kolommen na pivoteren: {df_pivot.columns.tolist()}")
        print("Controleer de waarden in de kolom '{GENDER_COLUMN_FOR_PIVOT}' in je Excel bestand (hoofdlettergevoeligheid!).")
        exit()

    # Zorg ervoor dat de gepivoteerde kolommen numeriek zijn en vul NaN met 0 (als een leeftijdscategorie geen mannen of vrouwen heeft)
    df_pivot[LEFT_SIDE_COLUMN_AFTER_PIVOT] = pd.to_numeric(df_pivot[LEFT_SIDE_COLUMN_AFTER_PIVOT], errors='coerce').fillna(0)
    df_pivot[RIGHT_SIDE_COLUMN_AFTER_PIVOT] = pd.to_numeric(df_pivot[RIGHT_SIDE_COLUMN_AFTER_PIVOT], errors='coerce').fillna(0)

    # Controleer of er nog steeds geldige numerieke waarden zijn voor plotting
    if df_pivot[LEFT_SIDE_COLUMN_AFTER_PIVOT].sum() == 0 and df_pivot[RIGHT_SIDE_COLUMN_AFTER_PIVOT].sum() == 0:
        print("\nWaarschuwing: Na pivoteren en omzetten naar numeriek, zijn alle waarden voor 'Man' en 'Vrouw' nul. De grafiek zal leeg zijn.")

    print("\nDataset na pivoteren. Eerste 5 rijen:")
    print(df_pivot.head())

except Exception as e:
    print(f"\nFout bij het pivoteren van de data: {e}")
    print(f"Controleer of de kolomnamen '{CATEGORY_COLUMN}', '{GENDER_COLUMN_FOR_PIVOT}', '{VALUE_COLUMN}' correct zijn en of je data geschikt is voor deze transformatie.")
    exit()


# --- Data voorbereiden voor de butterfly chart ---
# Sorteer de categorieën als 'Leeftijd' geen tekst is maar bijvoorbeeld 1-10, 11-20
# Als Leeftijd als tekst is opgeslagen zoals '0-4 jaar', '5-9 jaar' dan kan sortering lastig zijn.
# Indien nodig, moet hier een custom sorteerfunctie komen. Voor nu gaan we uit van de volgorde in de data.
categories = df_pivot[CATEGORY_COLUMN].tolist()
left_values = df_pivot[LEFT_SIDE_COLUMN_AFTER_PIVOT].values
right_values = df_pivot[RIGHT_SIDE_COLUMN_AFTER_PIVOT].values

# Maak de waarden voor de linkerkant negatief
left_values_negative = -left_values

# --- Butterfly Chart maken ---
fig, ax = plt.subplots(figsize=(10, 6))

# Teken de staven voor de linkerkant (Mannen)
ax.barh(categories, left_values_negative, color=LEFT_COLOR, label=LEFT_LABEL)

# Teken de staven voor de rechterkant (Vrouwen)
ax.barh(categories, right_values, color=RIGHT_COLOR, label=RIGHT_LABEL)

# Pas de X-as labels aan om absolute waarden te tonen
max_abs_val = max(abs(left_values).max(), abs(right_values).max())

# Genereer ticks en labels
num_ticks = 5 # Aantal ticks aan één kant (exclusief 0)
if max_abs_val == 0:
    step = 10 # Bijvoorbeeld, voor percentages, 10%
    x_ticks = np.array([-step, 0, step])
    x_labels = [f"{int(abs(tick))}%" for tick in x_ticks]
else:
    scale = 10**(np.floor(np.log10(max_abs_val / num_ticks)))
    step = np.ceil((max_abs_val / num_ticks) / scale) * scale
    if step == 0:
        step = 1
    elif step < 1: # Als stap bijv. 0.5 zou zijn, maak er 1 van voor percentages (visueel mooier)
        step = 1

    x_ticks = np.arange(-max_abs_val - step, max_abs_val + step, step)
    x_ticks = x_ticks[np.where(abs(x_ticks) <= max_abs_val * 1.1 + step)]
    x_ticks = np.unique(x_ticks)
    x_labels = [f"{int(abs(tick))}%" for tick in x_ticks]

# print(f"Debug Info: max_abs_val={max_abs_val}, calculated step={step}") # <-- Debug info

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