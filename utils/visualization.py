import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

BASE_DIR = "tex/imatges"  # Directori base per a les imatges

def guardar_grafica(filepath, fileformat='pdf', dpi=1200):
    """
    Guarda la gràfica actual al fitxer especificat dins del directori base.
    """
    full_path = os.path.join(BASE_DIR, filepath)

    dirpath = os.path.dirname(full_path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    plt.savefig(full_path, format=fileformat, dpi=dpi)
    print(f"Gràfica guardada a: {full_path}")

def grafiar_serie_temporal(data, title="Sèrie temporal", filepath=None, mostrar=True):
    """
    Mostra o guarda la gràfica de la sèrie temporal.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Sèrie temporal')
    plt.title(title)
    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.legend()
    if filepath:
        guardar_grafica(filepath)
    if mostrar:
        plt.show()
    plt.close()

def grafiar_prediccio(train, test, prediction, model_name, filepath=None, mostrar=True):
    """
    Mostra o guarda la gràfica de prediccions amb el nom del model.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train, label='Entrenament')
    plt.plot(test, label='Test')
    plt.plot(prediction, label='Predicció', linestyle='--')
    plt.title(f'Predicció de la sèrie temporal - Model {model_name}')
    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.legend()
    if filepath:
        guardar_grafica(filepath)
    if mostrar:
        plt.show()
    plt.close()

def grafiar_comparativa(comparativa, columna, model_name, filepath=None, mostrar=True):
    """
    Mostra o guarda la gràfica de comparació entre la predicció i el valor real, ajustant l'error perquè siga més visible.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Gràfica principal
    ax1.plot(comparativa[columna], label='Real', color='blue')
    ax1.plot(comparativa['Predicció'], label='Predicció', color='orange')
    ax1.set_ylabel('Valor')
    ax1.set_title(f'Comparació entre valors reals i prediccions - {model_name}')
    ax1.legend()

    # Gràfica de l'error (percentual o absolut)
    ax2.bar(comparativa.index, comparativa['Error (%)'], color='red', alpha=0.6, label='Error (%)')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Error (%)')
    ax2.set_xlabel('Data')
    ax2.legend()

    if filepath:
        guardar_grafica(filepath)
    if mostrar:
        plt.show()
    plt.close()

def grafiar_acf_pacf(data, lags=40, filepath=None, mostrar=True):
    """
    Mostra les gràfiques d'ACF i PACF, i opcionalment les guarda.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_acf(data, ax=axes[0], lags=lags)
    plot_pacf(data, ax=axes[1], lags=lags)
    axes[0].set_title("Funció d'autocorrelació (ACF)")
    axes[1].set_title("Funció d'autocorrelació parcial (PACF)")
    plt.tight_layout()
    if filepath:
        guardar_grafica(filepath)
    if mostrar:
        plt.show()
    plt.close(fig)

def grafiar_descomposicio(data, model='additive', freq=None, filepath=None, mostrar=True):
    """
    Mostra la descomposició de la sèrie temporal en components.
    """
    decomposition = seasonal_decompose(data, model=model, period=freq)
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    if filepath:
        guardar_grafica(filepath)
    if mostrar:
        plt.show()
    plt.close(fig)

def grafiar_boxplot_dia(data, columna, title="Box plot per dies de la setmana", filepath=None, mostrar=True):
    """
    Mostra un box plot per dies de la setmana.
    """
    if 'dia' not in data.columns:
        print("La columna 'dia' no existeix. S'ignora la generació del box plot per dies.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    data.boxplot(column=columna, by='dia', patch_artist=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Dia")
    ax.set_ylabel("Valor")
    ax.tick_params(axis='x', rotation=45)
    plt.suptitle("")
    if filepath:
        guardar_grafica(filepath)
    if mostrar:
        plt.show()
    plt.close(fig)

def grafiar_boxplot_mes(data, columna, title="Box plot per mesos", filepath=None, mostrar=True):
    """
    Mostra un box plot per mesos de l'any.
    """
    if 'mes' not in data.columns:
        print("La columna 'mes' no existeix. S'ignora la generació del box plot per mesos.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    data.boxplot(column=columna, by='mes', patch_artist=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Valor")
    ax.tick_params(axis='x', rotation=45)
    plt.suptitle("")
    if filepath:
        guardar_grafica(filepath)
    if mostrar:
        plt.show()
    plt.close(fig)

def grafiar_histograma_residus(residuals, title="Histograma dels residus", filepath=None, mostrar=True):
    """
    Mostra o guarda el histograma dels residus amb l'ajust a una distribució normal.

    :param residuals: Residus de la descomposició de la sèrie temporal
    :param title: Títol de la gràfica
    :param filepath: Ruta per guardar la imatge (opcional)
    :param mostrar: Si True, mostra la imatge; si False, només la guarda
    """
    residuals = residuals[~np.isnan(residuals)]

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, density=True, alpha=0.6, color='g', label="Residus")

    # Ajust a una distribució normal
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r', label="Normal teòrica")

    plt.title(title)
    plt.xlabel('Valor')
    plt.ylabel('Densitat')
    plt.legend()

    if filepath:
        guardar_grafica(filepath)
    if mostrar:
        plt.show()
    plt.close()

def grafiar_qqplot_residus(residuals, title="Gràfica Q-Q dels residus", filepath=None, mostrar=True):
    """
    Mostra o guarda la gràfica Q-Q dels residus per comprovar la normalitat.

    :param residuals: Residus de la descomposició de la sèrie temporal
    :param title: Títol de la gràfica
    :param filepath: Ruta per guardar la imatge (opcional)
    :param mostrar: Si True, mostra la imatge; si False, només la guarda
    """
    residuals = residuals[~np.isnan(residuals)]

    plt.figure(figsize=(8, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(title)

    if filepath:
        guardar_grafica(filepath)
    if mostrar:
        plt.show()
    plt.close()