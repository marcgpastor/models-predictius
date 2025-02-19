from pmdarima import auto_arima
import time

def ajustar_auto_arima(train, m=1):
    """
    Ajusta un model ARIMA automàtic provant tots els valors de d i D fins als màxims definits.

    Arguments:
    - train: Sèrie temporal d'entrenament.
    - m: Període d'estacionalitat.

    Retorna:
    - El millor model ARIMA segons AIC.
    """

    max_d = 2
    max_D = 2

    millor_model = None
    millor_aic = float("inf")
    millor_ordre = None
    millor_ordre_estacional = None

    print("Iniciant la cerca del millor model ARIMA...")

    for d in range(0, max_d + 1):
        for D in range(0, max_D + 1):
            print(f"Provant model amb d={d}, D={D}...")
            try:
                inici_temps = time.time()
                model = auto_arima(
                    train,
                    start_p=0, start_q=0,
                    max_p=2, max_q=2,
                    d=d, start_P=0, D=D, start_Q=0,
                    max_P=1, max_Q=1,
                    m=m, seasonal=True,
                    error_action='warn',
                    trace=True,
                    suppress_warnings=True,
                    stepwise=False,
                    random_state=20,
                    n_fits=50
                )
                temps_execucio = time.time() - inici_temps

                aic = model.aic()
                ordre = model.order
                ordre_estacional = model.seasonal_order

                print(f"El model amb d={d}, D={D} té un AIC de {aic:.3f} | Temps={temps_execucio:.2f} segons")

                if aic < millor_aic:
                    millor_aic = aic
                    millor_model = model
                    millor_ordre = ordre
                    millor_ordre_estacional = ordre_estacional

            except Exception as e:
                print(f"S'ha produït un error amb d={d}, D={D}: {e}")
                continue

    print(f"Model òptim seleccionat: ARIMA{millor_ordre}{millor_ordre_estacional}[{m}] | AIC={millor_aic:.3f}")

    return millor_model