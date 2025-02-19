import pandas as pd

def carregar_dades(filepath, freq='D', fill_method='ffill'):
    """
    Carrega el dataset des d'un fitxer CSV i prepara l'índex com a data.

    Parameters:
        filepath (str): Ruta del fitxer CSV.
        freq (str): Freqüència que s'assignarà al `DatetimeIndex` (per defecte, 'D' per diari).
        fill_method (str): Mètode per gestionar valors nuls ('ffill', 'bfill', 'interpolate', o 'drop').

    Returns:
        pd.DataFrame: DataFrame amb l'índex configurat com a `DatetimeIndex` únic i freqüència assignada.
    """
    try:
        dades = pd.read_csv(filepath, parse_dates=['data'], index_col='data')
    except FileNotFoundError:
        raise FileNotFoundError(f"El fitxer '{filepath}' no existeix.")
    except ValueError as e:
        raise ValueError(f"Error en carregar el fitxer '{filepath}': {e}")

    if 'data' not in dades.columns and not isinstance(dades.index, pd.DatetimeIndex):
        raise ValueError("El dataset no conté una columna de dates vàlida.")

    if dades.index.duplicated().any():
        print("S'han trobat duplicats a l'índex. S'estan eliminant...")
        dades = dades[~dades.index.duplicated(keep='first')]

    if freq == "ME":
        dades.index = dades.index + pd.offsets.MonthEnd(0)

    try:
        dades = dades.asfreq(freq)
    except ValueError:
        raise ValueError(f"No es pot assignar la freqüència '{freq}' perquè falten dates al dataset.")

    if fill_method == 'ffill':
        dades = dades.ffill().infer_objects(copy=False)
    elif fill_method == 'bfill':
        dades = dades.fillna(method='bfill').infer_objects(copy=False)
    elif fill_method == 'interpolate':
        dades = dades.interpolate(method='linear')
    elif fill_method == 'drop':
        dades = dades.dropna()
    else:
        raise ValueError(f"El mètode '{fill_method}' per gestionar valors nuls no és vàlid.")

    print(f"Dades carregades des de '{filepath}'. Rang de dates: {dades.index.min()} a {dades.index.max()}. Total registres: {len(dades)}.")
    return dades

def dividir_dades(dades, proporcio=0.8):
    """
    Divideix les dades en entrenament i test.

    Parameters:
        dades (pd.DataFrame): DataFrame complet a dividir.
        proporcio (float): Proporció de dades per a entrenament (per defecte: 0.8).

    Returns:
        pd.DataFrame, pd.DataFrame: Conjunts d'entrenament i test.
    """
    # Depuració: Comprovar longitud del dataset
    print(f"Longitud total del dataset: {len(dades)}")
    print(f"Proporció d'entrenament: {proporcio}")

    # Calcular la mida del conjunt d'entrenament
    train_size = int(len(dades) * proporcio)
    print(f"Mida del conjunt d'entrenament: {train_size}")
    print(f"Mida del conjunt de test: {len(dades) - train_size}")

    # Dividir les dades
    train = dades.iloc[:train_size]
    test = dades.iloc[train_size:]

    # Comprovar valors nuls en cada conjunt
    print("-" * 50)
    print("Comprovació de valors nuls:")
    print(f"Entrenament - NaN: {train.isnull().sum().sum()}")
    print(f"Test - NaN: {test.isnull().sum().sum()}")

    return train, test

def afegir_ordre_temporal(dades):
    """
    Afegeix columnes 'dia' i 'mes' amb ordre categòric.
    """
    ordre_dies = ['dilluns', 'dimarts', 'dimecres', 'dijous', 'divendres', 'dissabte', 'diumenge']
    ordre_mesos = ['gener', 'febrer', 'març', 'abril', 'maig', 'juny',
                   'juliol', 'agost', 'setembre', 'octubre', 'novembre', 'desembre']

    if 'dia' in dades.columns:
        dades['dia'] = pd.Categorical(dades['dia'], categories=ordre_dies, ordered=True)
    if 'mes' in dades.columns:
        dades['mes'] = pd.Categorical(dades['mes'], categories=ordre_mesos, ordered=True)

    return dades

def seleccionar_columnes(dades, CONFIG):
    columna = CONFIG["columna"]
    columnes_opcionals = ['any', 'mes', 'dia']
    columnes_existents = [col for col in columnes_opcionals if col in dades.columns]
    columnes_final = [columna] + columnes_existents
    dades = dades[columnes_final]
    return dades, columna

def filtrar_dades(dades, CONFIG):
    n = int(len(dades) * CONFIG["proporcio_dataset"])
    dades = dades.iloc[-n:]
    print(f"Rang de dates: {dades.index.min()} a {dades.index.max()}")
    print(f"Número de registres: {len(dades)} ({CONFIG['proporcio_dataset']:.0%})")
    return dades