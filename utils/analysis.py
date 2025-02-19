import os
import pandas as pd
import numpy as np
from pmdarima.arima import ADFTest
from pmdarima.arima.utils import nsdiffs
from scipy.stats import shapiro
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.tools import diff
from statsmodels.stats.stattools import jarque_bera

def test_estacionarietat(data, alpha=0.05):
    """
    Realitza el test Augmented Dickey-Fuller per comprovar l'estacionarietat.
    """
    adf_test = ADFTest(alpha=alpha)
    d = 0
    print(f"Comprovant estacionarietat amb d={d}")
    p_valor, necessita_diferenciar = adf_test.should_diff(data)
    print(f"p-valor: {p_valor}")
    print(f"Necessita diferenciació: {necessita_diferenciar}")

    while necessita_diferenciar:
        d += 1
        print("-" * 50)
        print(f"Comprovant estacionarietat amb d={d}")
        data = data.diff().dropna()
        p_valor, necessita_diferenciar = adf_test.should_diff(data)
        print(f"p-valor: {p_valor}")
        print(f"Necessita diferenciació: {necessita_diferenciar}")

    return d


import numpy as np
import pandas as pd
from scipy.stats import kruskal


def test_estacionarietat_estacional(data, m, alpha=0.05):
    """
    Realitza el test de Kruskal-Wallis per comprovar si la component estacional és significativa.

    Paràmetres:
    - data: pandas Series amb la sèrie temporal.
    - periode: int, el període estacional a comprovar (per exemple, 12 per dades mensuals).
    - alpha: nivell de significació per al test de Kruskal-Wallis (per defecte, 0.05).

    Retorna:
    - D: Nombre de diferenciacions estacionals necessàries.
    """
    D = 0
    print(f"Comprovant estacionarietat estacional amb D={D}")

    # Separar les dades per cicles estacionals
    grups = [data[i::m] for i in range(m)]

    # Aplicar el test de Kruskal-Wallis
    h_stat, p_valor = kruskal(*grups)
    print(f"Estadístic H: {h_stat}, p-valor: {p_valor}")

    while p_valor < alpha:
        D += 1
        print("-" * 50)
        print(f"Comprovant estacionarietat estacional amb D={D}")
        data = data.diff(m).dropna()  # Diferenciació estacional
        grups = [data[i::m] for i in range(m)]

        if any(len(g) < 5 for g in grups):  # Verifiquem si tenim suficients dades per cada grup
            print("No hi ha suficients dades per continuar el test.")
            break

        h_stat, p_valor = kruskal(*grups)
        print(f"Estadístic H: {h_stat}, p-valor: {p_valor}")

    return D

def diferenciar_serie(data, m=1, d=0, D=0):
    """
    Aplica diferenciació a la sèrie temporal i retorna un DataFrame.

    :param data: Sèrie temporal (pandas Series).
    :param m: Període estacional.
    :param d: Nombre de diferenciacions regulars.
    :param D: Nombre de diferenciacions estacionals.
    :return: DataFrame amb la sèrie diferenciada.
    """
    data_dif = diff(
        series=data,
        k_diff=d,
        k_seasonal_diff=D,
        seasonal_periods=m
    )

    # Convertir en DataFrame i preservar l'índex original ajustat
    df_dif = pd.DataFrame(data_dif, index=data.index[max(d, D) * m:], columns=[data.name])

    return df_dif

def calcular_metriques(real, prediccio):
    """
    Calcula les mètriques d'error entre els valors reals i les prediccions.

    Parameters:
        real (pd.Series): Valors reals.
        prediccio (pd.Series): Prediccions.

    Returns:
        dict: Diccionari amb RMSE i MAPE.
    """
    # Comprovar que les sèries tenen la mateixa longitud
    if len(real) != len(prediccio):
        raise ValueError("Les sèries real i predicció han de tindre la mateixa longitud.")

    rmse = np.sqrt(np.mean((real - prediccio) ** 2))
    mape = np.mean(np.abs((real - prediccio) / real)) * 100

    return {"RMSE": rmse, "MAPE": mape}

def guardar_taula_metriques(metriques, filepath="tex/altres/metriques.csv"):
    """
    Genera una taula de mètriques i la desa com a CSV.

    Parameters:
        metriques (dict): Diccionari amb les mètriques per model.
        filepath (str): Ruta on desar el CSV.

    Returns:
        pd.DataFrame: Taula de mètriques.
    """
    # Convertir les mètriques a DataFrame
    df = pd.DataFrame(metriques).T
    df.columns = ["RMSE", "MAPE"]

    # Desa el resultat com a CSV
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, float_format="%.2f")

    return df

def descomposicio_estacional(data, freq=12):
    """
    Descomposició de la sèrie temporal en components estacionaris.

    Parameters:
        data (pd.Series): Sèrie temporal.
        freq (int): Freqüència de la sèrie.

    Returns:
        statsmodels.tsa.seasonal.DecomposeResult: Resultat de la descomposició.
    """
    return seasonal_decompose(
        data,
        model='additive',
        period=freq,
        two_sided=True
    )

def pes_tendencia(descomposicio):
    """
    Calcula el pes de la tendència en la sèrie temporal.

    Parameters:
        descomposicio (statsmodels.tsa.seasonal.DecomposeResult): Resultat de la descomposició.

    Returns:
        float: Pes de la tendència.
    """
    return max(0, 1 - (np.var(descomposicio.resid)/np.var(descomposicio.trend + descomposicio.resid)))

def pes_estacionalitat(descomposicio):
    """
    Calcula el pes de l'estacionalitat en la sèrie temporal.

    Parameters:
        descomposicio (statsmodels.tsa.seasonal.DecomposeResult): Resultat de la descomposició.

    Returns:
        float: Pes de l'estacionalitat.
    """
    return max(0, 1 - (np.var(descomposicio.resid)/np.var(descomposicio.seasonal + descomposicio.resid)))

def coeficient_r2(y, y_pred):
    """
    Calcula el coeficient de determinació R^2.

    Parameters:
        y (pd.Series): Valors reals.
        y_pred (pd.Series): Prediccions.

    Returns:
        float: Coeficient de determinació R^2.
    """
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def descriptiva(dades, columna, CONFIG):
    describe_general = dades.drop(columns=['any']).describe()
    describe_mesos = dades.groupby("mes", observed=False)[columna].describe()

    print(describe_general)
    print("-" * 50)
    print(describe_mesos)

    print("-" * 50)
    describe_general.to_csv(f"{CONFIG.get("others_path")}/descriptiva_general.csv")
    describe_mesos.to_csv(f"{CONFIG.get("others_path")}/descriptiva_per_mesos.csv")
    print("Descriptives exportades correctament.")

def test_jarque_bera(residuals):
    """
    Realitza el test de Jarque-Bera per comprovar la normalitat dels residus.

    :param residuals: Residus de la descomposició de la sèrie temporal
    """
    residuals = residuals[~np.isnan(residuals)]  # Elimina NaNs
    jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)

    print(f"Jarque-Bera test: estadístic={jb_stat:.4f}, p-valor={jb_p:.4f}")
    print(f"Asimetria (skewness): {skew:.4f}")
    print(f"Excés de curtosi: {kurtosis:.4f}")

    if jb_p > 0.05:
        print("No es pot rebutjar la hipòtesi de normalitat (p-valor alt).")
        return True  # Els residus podrien ser normals
    else:
        print("Es rebutja la hipòtesi de normalitat (p-valor baix).")
        return False  # Els residus no són normals

def test_shapiro_wilk(residuals):
    """
    Realitza el test de Shapiro-Wilk per comprovar la normalitat dels residus.

    :param residuals: Residus de la sèrie descomposta
    """
    residuals = residuals[~np.isnan(residuals)]  # Elimina NaNs
    sw_stat, sw_p = shapiro(residuals)
    print(f"Shapiro-Wilk test: estadístic={sw_stat:.4f}, p-valor={sw_p:.4f}")

    if sw_p > 0.05:
        print("No es pot rebutjar la hipòtesi de normalitat (p-valor alt).")
    else:
        print("Es rebutja la hipòtesi de normalitat (p-valor baix).")

def taula_comparativa(test, prediction, columna):
    comparativa = pd.concat([test, prediction], axis=1)
    comparativa['Diferència'] = comparativa[columna] - comparativa['Predicció']
    comparativa['Error (%)'] = round((comparativa['Diferència'] / comparativa[columna]) * 100, 2)

    return comparativa