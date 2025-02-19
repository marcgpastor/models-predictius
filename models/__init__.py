from .auto_arima import ajustar_auto_arima
from .arima import ajustar_arima
from .holt_winters import ajustar_holt_winters
from .prophet import ajustar_prophet, predir_prophet

def obtindre_model(config):
    return {
        "AUTO-ARIMA": lambda train: ajustar_auto_arima(train, m=config["m"]),
        "ARIMA": lambda train: ajustar_arima(train, m=config["m"]),
        "Holt-Winters": lambda train: ajustar_holt_winters(train, seasonal="add", seasonal_periods=config["m"]),
        "Prophet": lambda train: ajustar_prophet(train, m=config["m"]),
    }

def obtindre_prediccio():
    return {
        "Prophet": lambda model, n_periods, freq, test_index: predir_prophet(model, n_periods, freq, test_index),
        "Holt-Winters": lambda model, n_periods, *_: model.forecast(steps=n_periods),
        "AUTO-ARIMA": lambda model, n_periods, *_: model.predict(n_periods=n_periods),
        "ARIMA": lambda model, n_periods, *_: model.predict(n_periods=n_periods),
    }