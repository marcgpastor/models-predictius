from statsmodels.tsa.holtwinters import ExponentialSmoothing

def ajustar_holt_winters(train, seasonal="add", seasonal_periods=1, trend="add", damped_trend=False):
    """
    Ajusta un model Holt-Winters (Suavització Exponencial).

    Arguments:
    - train: Sèrie temporal d'entrenament.
    - seasonal: Tipus d'estacionalitat ("add" o "mul").
    - seasonal_periods: Període d'estacionalitat.
    - trend: Tipus de tendència ("add", "mul" o None).
    - damped_trend: Indica si la tendència ha d'estar esmorteïda.

    Retorna:
    - Model Holt-Winters ajustat o None si hi ha un error.
    """
    try:
        model = ExponentialSmoothing(
            train,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            trend=trend,
            damped_trend=damped_trend
        ).fit(optimized=True, use_boxcox=None, remove_bias=True)

        print(
            f"Model Holt-Winters ajustat correctament amb estacionalitat {seasonal}, període {seasonal_periods}, tendència {trend} i damped_trend {damped_trend}.")
        return model

    except Exception as e:
        print(f"S'ha produït un error en ajustar Holt-Winters: {e}")
        return None