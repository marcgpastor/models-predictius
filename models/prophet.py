from prophet import Prophet
import pandas as pd

def ajustar_prophet(train, m=1, d=0):
    """
    Ajusta un model Prophet amb estacionalitat segons el valor de m i diferenciació si cal.

    Arguments:
    - train: Sèrie temporal d'entrenament.
    - m: Període d'estacionalitat.
    - d: Nombre de diferenciacions aplicades abans de l'entrenament.

    Retorna:
    - Model Prophet ajustat.
    """
    df_train = train.reset_index().rename(columns={train.index.name: "ds", train.columns[0]: "y"})

    model = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.05)

    if m == 7:
        model.add_seasonality(name="daily", period=7, fourier_order=5)
    elif m == 12:
        model.add_seasonality(name="monthly", period=12, fourier_order=15)
    elif m == 52:
        model.add_seasonality(name="weekly", period=52, fourier_order=20)

    model.fit(df_train)

    return model

def predir_prophet(model, periods, freq="M", train=None, d=0):
    """
    Genera prediccions amb un model Prophet i assegura que es corresponen amb el test.
    Si s'ha aplicat diferenciació (d > 0), es reintegra la predicció a l'escala original.

    Arguments:
    - model: Model Prophet ajustat.
    - periods: Nombre de períodes a predir.
    - freq: Freqüència de la predicció (per defecte "M" per mensual).
    - train: Conjunt d'entrenament original (necessari per a reintegració si d > 0).
    - d: Nombre de diferenciacions aplicades.

    Retorna:
    - Sèrie de prediccions amb l'índex corregit.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)

    forecast = model.predict(future)
    forecast["ds"] = pd.to_datetime(forecast["ds"])

    prediccions = forecast.set_index("ds")["yhat"].iloc[-periods:]
    prediccions.index.name = "data"
    prediccions.index = prediccions.index + pd.offsets.MonthEnd(0)

    if d > 0 and train is not None:
        ultim_valor_train = train.iloc[-1, 0]
        prediccions = prediccions.cumsum() + ultim_valor_train

    return prediccions.rename("Predicció")