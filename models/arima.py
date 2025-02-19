from pmdarima.arima import ARIMA
import itertools
import time

def ajustar_arima(train, p_range=(0,2), d_range=(0,2), q_range=(0,2), P_range=(0,1), D_range=(0,1), Q_range=(0,1), m=12):
    """
    Ajusta un model ARIMA provant diferents valors dels paràmetres i seleccionant el millor segons AIC.

    Arguments:
    - train: Sèrie temporal d'entrenament.
    - p_range, d_range, q_range: Rangs per als paràmetres ARIMA.
    - P_range, D_range, Q_range: Rangs per als paràmetres estacionals SARIMA.
    - m: Periodicitat estacional.

    Retorna:
    - El millor model ARIMA segons AIC.
    """

    millor_model = None
    millor_aic = float("inf")
    millor_ordre = None
    millor_ordre_estacional = None

    print("Realitzant cerca pas a pas per minimitzar l'AIC")

    combinacions_parametres = list(itertools.product(range(p_range[0], p_range[1] + 1),
                                                     range(d_range[0], d_range[1] + 1),
                                                     range(q_range[0], q_range[1] + 1)))
    combinacions_estacionals = list(itertools.product(range(P_range[0], P_range[1] + 1),
                                                      range(D_range[0], D_range[1] + 1),
                                                      range(Q_range[0], Q_range[1] + 1)))

    for ordre in combinacions_parametres:
        for ordre_estacional in combinacions_estacionals:
            try:
                inici_temps = time.time()
                model = ARIMA(
                    order=ordre,
                    seasonal_order=ordre_estacional + (m,),
                    suppress_warnings=True
                ).fit(train)
                temps_execucio = time.time() - inici_temps

                aic = model.aic()
                ordre_str = f"ARIMA{ordre}{ordre_estacional + (m,)}"
                temps_str = f"Temps={temps_execucio:.2f} segons"
                print(f" {ordre_str:<35}: AIC={aic:.3f}, {temps_str}")

                if aic < millor_aic:
                    millor_aic = aic
                    millor_model = model
                    millor_ordre = ordre
                    millor_ordre_estacional = ordre_estacional

            except Exception as e:
                print(f"Error amb ARIMA{ordre}{ordre_estacional + (m,)}: {e}")
                continue

    print(f"\nMillor model seleccionat: ARIMA{millor_ordre} Seasonal{millor_ordre_estacional + (m,)} | AIC={millor_aic:.3f}")

    return millor_model