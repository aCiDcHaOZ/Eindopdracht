import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import cartopy.io.shapereader as shpreader

# --- Data Laden en Filteren (overgenomen van het vorige Canvas) ---
# Pad naar je Excel-bestand
excel_file_path = 'dataset.xlsx'
sheet_name = 'Tabel 1'

try:
    # Laad de specifieke tabel uit het Excel-bestand
    # Stel 'header=3' in om de 4e regel (index 3) als kolomnamen te gebruiken
    df_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name, header=3)
    print(f"Origineel DataFrame geladen uit '{sheet_name}':")
    print(df_raw.head())
    print("-" * 30)

    # Print alle kolomnamen om te helpen bij het debuggen
    print("Kolomnamen in het geladen DataFrame:")
    print(df_raw.columns.tolist())
    print("-" * 30)

    # Definieer de filtercriteria
    filter_kenmerk = 'Kenmerk'
    filter_type_personeel = 'Type personeel'
    filter_defensieonderdeel = 'Defensieonderdeel'
    filter_jaar = 'Jaar'

    # Filter het DataFrame met de nieuwe criteria
    df_filtered = df_raw[
        (df_raw[filter_kenmerk] == 'Herkomst_nieuw') &
        (df_raw[filter_type_personeel] == 'Totaal') &
        (df_raw[filter_defensieonderdeel] == 'Totaal') &
        (df_raw[filter_jaar] == 2022) &
        (df_raw['Categorie'] != 'Nederland') # Nieuw filter: verwijder 'Nederland'
    ].copy()

    print(f"Gefilterd DataFrame (Kenmerk='Herkomst_nieuw', Type personeel='Totaal', Defensieonderdeel='Totaal', Jaar='2022', Excl. Nederland):")
    print(df_filtered.head())
    print(f"\nAantal rijen in gefilterd DataFrame: {len(df_filtered)}")
    print("-" * 30)

    # Controleer of de benodigde kolommen aanwezig zijn
    required_columns = ['Categorie', 'Aantal']
    if not all(col in df_filtered.columns for col in required_columns):
        print(f"Waarschuwing: Niet alle benodigde kolommen ({', '.join(required_columns)}) zijn aanwezig in het gefilterde DataFrame.")
        print("Beschikbare kolommen:", df_filtered.columns.tolist())

except FileNotFoundError:
    print(f"Fout: Het bestand '{excel_file_path}' is niet gevonden. Zorg ervoor dat het bestand in de juiste map staat.")
    # Maak een leeg DataFrame voor demonstratie als het bestand niet wordt gevonden
    # Voeg hier handmatig voorbeelddata toe voor demonstratie
    df_filtered = pd.DataFrame({
        'Categorie': ['België', 'Turkije', 'Europa (Exclusief Nederland)', 'Overig Afrika, Azië, Amerika en Oceanië'],
        'Aantal': [150, 80, 700, 200] # Voorbeeldwaarden (Nederland is hier al uitgesloten)
    })
    print("Leeg DataFrame gemaakt voor demonstratie met voorbeelddata. De kaart zal nu data tonen.")
except KeyError as e:
    print(f"Fout: De opgegeven kolom '{e}' is niet gevonden. Controleer de kolomnamen en de sheetnaam.")
    df_filtered = pd.DataFrame(columns=['Categorie', 'Aantal'])
    print("Leeg DataFrame gemaakt voor demonstratie. De kaart zal leeg zijn.")
except Exception as e:
    print(f"Er is een onverwachte fout opgetreden: {e}")
    df_filtered = pd.DataFrame(columns=['Categorie', 'Aantal'])
    print("Leeg DataFrame gemaakt voor demonstratie. De kaart zal leeg zijn.")

# --- Voorbeeld mapping van 'Categorie' naar landnamen die Cartopy begrijpt ---
# BELANGRIJK: Pas deze mapping aan op basis van de daadwerkelijke waarden in je 'Categorie' kolom
# en de landnamen die Cartopy gebruikt (meestal de officiële Engelse namen).
# 'Europa (Exclusief Nederland)' en 'Overig Afrika, Azië, Amerika en Oceanië' worden apart behandeld.
country_mapping = {
    # 'Nederland': 'Netherlands', # Verwijderd uit mapping omdat het wordt gefilterd
    'België': 'Belgium',
    'Duitsland': 'Germany',
    'Frankrijk': 'France',
    'Verenigd Koninkrijk': 'United Kingdom',
    'Spanje': 'Spain',
    'Italië': 'Italy',
    'Polen': 'Poland',
    'Zweden': 'Sweden',
    'Noorwegen': 'Norway',
    'Denemarken': 'Denmark',
    'Oostenrijk': 'Austria',
    'Zwitserland': 'Switzerland',
    'Portugal': 'Portugal',
    'Ierland': 'Ireland',
    'Griekenland': 'Greece',
    'Turkije': 'Turkey',
    'Marokko': 'Morocco',
    'Suriname': 'Suriname',
    'Indonesië': 'Indonesia',
    # Voeg hier meer mappings toe voor alle relevante landen in je 'Categorie' kolom
    # Let op: 'Nederlands-caribisch gebied' moet mogelijk individuele eilanden mappen
    # als je die wilt inkleuren.
}

# Lijst van Europese landen (exclusief Nederland) die de 'Europa (Exclusief Nederland)' waarde krijgen
# Deze lijst moet overeenkomen met de 'NAME_LONG' attributen van Natural Earth.
european_countries_excluding_netherlands = [
    'Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina',
    'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland',
    'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy',
    'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta',
    'Moldova', 'Monaco', 'Montenegro', 'North Macedonia', 'Norway', 'Poland',
    'Portugal', 'Romania', 'Russia', # Rusland is deels Europees, afhankelijk van definitie
    'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden',
    'Switzerland', 'Ukraine', 'United Kingdom', 'Vatican City'
]

# Haal de waarde voor 'Europa (Exclusief Nederland)' op, indien aanwezig
europa_excl_nl_value = df_filtered[df_filtered['Categorie'] == 'Europa (Exclusief Nederland)']['Aantal'].iloc[0] \
    if 'Europa (Exclusief Nederland)' in df_filtered['Categorie'].values else None
print(f"Waarde voor 'Europa (Exclusief Nederland)': {europa_excl_nl_value}")

# Haal de waarde voor 'Overig Afrika, Azië, Amerika en Oceanië' op, indien aanwezig
overig_wereld_value = df_filtered[df_filtered['Categorie'] == 'Overig Afrika, Azië, Amerika en Oceanië']['Aantal'].iloc[0] \
    if 'Overig Afrika, Azië, Amerika en Oceanië' in df_filtered['Categorie'].values else None
print(f"Waarde voor 'Overig Afrika, Azië, Amerika en Oceanië': {overig_wereld_value}")


# Voeg een kolom 'Landnaam_Cartopy' toe aan df_filtered
df_filtered['Landnaam_Cartopy'] = df_filtered['Categorie'].map(country_mapping)
print("\nDataFrame na mapping van 'Categorie' naar 'Landnaam_Cartopy':")
print(df_filtered[['Categorie', 'Landnaam_Cartopy', 'Aantal']].head(10))
print("-" * 30)


# Zorg ervoor dat 'Aantal' numeriek is en behandel eventuele niet-numerieke waarden
df_filtered['Aantal'] = pd.to_numeric(df_filtered['Aantal'], errors='coerce').fillna(0)

# --- Cartopy Heatmap Plot ---
if not df_filtered.empty and 'Landnaam_Cartopy' in df_filtered.columns:
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_global()

    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5, facecolor='blue')
    ax.add_feature(cfeature.RIVERS)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    shpfilename = shpreader.natural_earth(resolution='50m',
                                          category='cultural',
                                          name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)

    if not df_filtered['Aantal'].empty:
        # Combineer alle waarden voor normalisatie, inclusief de geaggregeerde
        all_values_for_norm = df_filtered['Aantal'].tolist()
        if europa_excl_nl_value is not None:
            all_values_for_norm.append(europa_excl_nl_value)
        if overig_wereld_value is not None:
            all_values_for_norm.append(overig_wereld_value)

        min_val = min(all_values_for_norm) if all_values_for_norm else 0
        max_val = max(all_values_for_norm) if all_values_for_norm else 1

        if max_val == min_val:
            norm = Normalize(vmin=min_val - 1, vmax=min_val + 1)
        else:
            norm = Normalize(vmin=min_val, vmax=max_val)

        # Definieer een colormap: van geel naar rood (YlOrRd)
        cmap = plt.cm.YlOrRd

        # Houd bij welke landen al zijn ingekleurd door specifieke data
        countries_colored_by_specific_data = set(df_filtered['Landnaam_Cartopy'].dropna().tolist())

        for country_record in reader.records():
            country_geom = country_record.geometry
            country_name = country_record.attributes['NAME_LONG']
            continent_name = country_record.attributes.get('CONTINENT') # Gebruik .get() om fouten te voorkomen

            value_to_plot = None
            color = None # Initialiseer kleur

            # Debugging: print de landnaam en continent
            # print(f"Verwerken land: {country_name} (Continent: {continent_name})")

            if country_name == 'Netherlands': # Nederland wordt nu expliciet overgeslagen
                continue # Sla dit land over in de plot

            if country_name in countries_colored_by_specific_data:
                # Prioriteit 1: Specifieke data voor dit land
                value_to_plot = df_filtered[df_filtered['Landnaam_Cartopy'] == country_name]['Aantal'].iloc[0]
                # print(f"  -> Specifieke data gevonden: {value_to_plot}")
            elif country_name in european_countries_excluding_netherlands and europa_excl_nl_value is not None:
                # Prioriteit 2: Geaggregeerde data voor 'Europa (Exclusief Nederland)'
                value_to_plot = europa_excl_nl_value
                # print(f"  -> Europa (Excl. NL) data gebruikt: {value_to_plot}")
            elif continent_name in ['Africa', 'Asia', 'North America', 'South America', 'Oceania'] and \
                 country_name not in countries_colored_by_specific_data and \
                 overig_wereld_value is not None:
                # Prioriteit 3: Overige wereld data voor niet-specifieke landen buiten Europa (excl. NL)
                value_to_plot = overig_wereld_value
                # print(f"  -> Overig Wereld data gebruikt: {value_to_plot}")

            if value_to_plot is not None:
                color = cmap(norm(value_to_plot))
                ax.add_geometries([country_geom], ccrs.PlateCarree(), facecolor=color, edgecolor='black', linewidth=0.5)
            # else:
                # print(f"  -> Geen data voor dit land, blijft standaard kleur.")


        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Intensiteit (Aantal)')

    plt.title('Herkomst_nieuw Totaal Aantal per Land Wereldwijd')
    plt.show()
else:
    print("Geen data om te plotten na filtering of ontbrekende 'Landnaam_Cartopy' kolom.")