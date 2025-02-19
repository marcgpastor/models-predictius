import os
import pandas as pd
from utils import analysis, utils, preprocessing as prep, visualization as visual
from models import obtindre_model, obtindre_prediccio

CONFIG = {
    "dataset_path": "data/passatgers.csv",
    "graphics_path": os.path.abspath("tex/imatges"),
    "others_path": os.path.abspath("tex/altres"),
    "freq": "ME",
    "columna": "nacional",
    "m": 12, # Opcions: 1, 7, 12, 52
    "proporcio_dataset": 1,
    "proporcio_train": 0.95,
}

SECCIONS = {
    "descriptiva": True,
    "grafiques": {
        "serie_temporal": True,
        "acf_pacf": True,
        "descomposicio": True,
        "histograma_residus": True,
        "qqplot_residus": True,
        "boxplot_mes": True,
        "prediccio": True,
        "comparativa": True,
    },
    "descomposicio": True,
    "soroll_blanc": True,
    "estacionarietat": True,
    "models": {
        "ARIMA": True,
        "AUTO-ARIMA": True,
        "Holt-Winters": True,
        "Prophet": True,
    },
    "resum": True,
    "metriques": True,
}

MODELS = obtindre_model(CONFIG)
PREDICCIONS = obtindre_prediccio()

if __name__ == "__main__":
    # CÀRREGA I FILTRACIÓ DE DADES
    print("=" * 50)
    print("CÀRREGA I FILTRACIÓ DE DADES")
    print("-" * 50)
    dades = prep.carregar_dades(CONFIG["dataset_path"], freq=CONFIG["freq"])
    dades = prep.afegir_ordre_temporal(dades)
    dades, columna = prep.seleccionar_columnes(dades, CONFIG)
    print("=" * 50)

    # INFORMACIÓ DE LES DADES SELECCIONADES
    print("INFORMACIÓ DE LES DADES SELECCIONADES")
    print("-" * 50)
    dades = prep.filtrar_dades(dades, CONFIG)
    print("-" * 50)
    visual.grafiar_serie_temporal(
        dades[[columna]],
        title=f"Sèrie temporal de '{columna}'",
        filepath="analisi/serie.pdf",
        mostrar=SECCIONS.get("grafiques", {}).get("serie_temporal", None)
    )
    print("=" * 50)

    # DESCRIPTIVA
    if SECCIONS["descriptiva"]:
        print("DESCRIPTIVA DE LES DADES")
        print("-" * 50)
        analysis.descriptiva(dades, columna, CONFIG)
        print("=" * 50)

    # DESCOMPOSICIÓ
    if SECCIONS["descomposicio"]:
        print("DESCOMPOSICIÓ")
        print("-" * 50)
        descomposicio = analysis.descomposicio_estacional(dades[columna], freq=CONFIG["m"])
        visual.grafiar_descomposicio(
            dades[columna],
            model='additive',
            freq=CONFIG.get("m", 12),
            filepath="analisi/descomposicio.pdf",
            mostrar=SECCIONS.get("grafiques", {}).get("descomposicio", None)
        )
        print("-" * 50)
        visual.grafiar_boxplot_mes(
            dades, columna,
            title=f"Box plot per mesos ({columna})",
            filepath="analisi/boxplot_mes.pdf",
            mostrar=SECCIONS.get("grafiques", {}).get("boxplot_mes", None)
        )
        print("-" * 50)

        # Força de la tendència i la estacionalitat
        print(f"Força de la tendència: {analysis.pes_tendencia(descomposicio):.2f}/1")
        print(f"Força de l'estacionalitat: {analysis.pes_estacionalitat(descomposicio):.2f}/1")
        print("=" * 50)

    # SOROLL BLANC
    if SECCIONS["descomposicio"] and SECCIONS["soroll_blanc"]:
        print("SOROLL BLANC")
        print("-" * 50)
        analysis.test_jarque_bera(descomposicio.resid)
        print("-" * 50)
        analysis.test_shapiro_wilk(descomposicio.resid)
        print("-" * 50)
        visual.grafiar_histograma_residus(
            descomposicio.resid,
            filepath="analisi/histograma_residus.pdf",
            mostrar=SECCIONS.get("grafiques", {}).get("histograma_residus", None)
        )
        visual.grafiar_qqplot_residus(
            descomposicio.resid,
            filepath="analisi/qqplot_residus.pdf",
            mostrar=SECCIONS.get("grafiques", {}).get("qqplot_residus", None)
        )
        print("=" * 50)

    # ESTACIONARIETAT I DIFERENCIACIÓ
    if SECCIONS["estacionarietat"]:
        print("ESTACIONARIETAT")
        print("-" * 50)
        d = analysis.test_estacionarietat(dades[columna])
        print("-" * 50)
        D = analysis.test_estacionarietat_estacional(dades[columna], m=CONFIG["m"])
        print("-" * 50)
        print(f"Ordre de diferenciació: d={d}, D={D}")
        dades_dif = analysis.diferenciar_serie(dades[columna], m=CONFIG["m"], d=d, D=D).dropna()
        visual.grafiar_acf_pacf(
            data = dades_dif[[columna]],
            lags = 40,
            filepath = "analisi/acf_pacf.pdf",
            mostrar = SECCIONS.get("grafiques", {}).get("acf_pacf", None)
        )
        print("=" * 50)

    # DIVISIÓ DE DADES
    print("DIVISIÓ DE DADES")
    print("-" * 50)
    train, test = prep.dividir_dades(dades[[columna]], proporcio=CONFIG["proporcio_train"])
    print("=" * 50)

    # MODELS
    taula_metriques = {}
    for model_name, active in SECCIONS["models"].items():
        if not active:
            continue

        print(f"MODEL {model_name.upper()}")
        print("-" * 50)
        dataset_name = os.path.splitext(os.path.basename(CONFIG["dataset_path"]))[0]
        model_name_m = f"{model_name.lower().replace(' ', '_')}_{CONFIG["m"]}"
        model_path = f"saved_models/{dataset_name}_{model_name_m}_{len(dades)}.pkl"

        # Entrena o carrega el model
        try:
            model = utils.carregar_model(model_path)
        except FileNotFoundError:
            print(f"Model {model_name} no trobat. Entrenant...")
            model = MODELS[model_name](train)
            utils.guardar_model(model, model_path)

        try:
            model_summary = model.summary()
        except AttributeError:
            model_summary = f"El model {model_name} no té un mètode summary()."

        # RESUM
        if SECCIONS.get("resum", False):
            print("-" * 50)
            print(model_summary)
            print("-" * 50)

        # PREDICCIONS
        n_periods = len(test)
        if model_name in PREDICCIONS:
            predicted = PREDICCIONS[model_name](model, n_periods, CONFIG["freq"], test.index)
        else:
            raise ValueError(f"Model {model_name} no implementat.")

        prediction = pd.DataFrame(predicted, index=test.index, columns=['Predicció'])
        prediction['Predicció'] = prediction['Predicció'].round(0).astype(int)
        visual.grafiar_prediccio(
            train[[columna]],
            test[[columna]],
            prediction,
            model_name,
            filepath=f"prediccions/prediccio_{model_name.lower().replace(' ', '_')}.pdf",
            mostrar=SECCIONS["grafiques"]["prediccio"]
        )
        print("=" * 50)

        # COMPARATIVA VALORS REALS I PREDICCIONS
        print("COMPARATIVA VALORS REALS I PREDICCIONS")
        print("-" * 50)
        comparativa = analysis.taula_comparativa(test, prediction, columna)
        print(comparativa)
        print("-" * 50)
        filepath = f"prediccions/error_{model_name.lower().replace(' ', '_')}.pdf"
        visual.grafiar_comparativa(comparativa, columna, model_name, filepath=filepath, mostrar=SECCIONS["grafiques"]["comparativa"])
        print("=" * 50)

        # CÀLCUL DE R^2
        print("CÀLCUL DE R^2")
        print("-" * 50)
        print(f"R^2: {analysis.coeficient_r2(test[columna], prediction['Predicció']):.2f}")
        print("=" * 50)

        # MÈTRIQUES
        metriques = analysis.calcular_metriques(test[columna], predicted)
        taula_metriques[model_name] = metriques

    # MÈTRIQUES
    if SECCIONS["metriques"]:
        print("MÈTRIQUES")
        print("-" * 50)
        print(pd.DataFrame(taula_metriques))
        analysis.guardar_taula_metriques(taula_metriques, filepath=f"{CONFIG.get("others_path")}/metriques.csv")
        print("=" * 50)