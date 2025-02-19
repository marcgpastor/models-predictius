# Projecte TFG - Anàlisi Predictiva de Sèries Temporals

Aquest projecte explora algoritmes i models aplicats a l'anàlisi de sèries temporals, incloent-hi els models SARIMA, Holt-Winters i Prophet.

L'estructura del projecte està dissenyada per a ser modular i fàcil de mantindre, amb funcions i mòduls separats per a cada tasca específica. 
El fitxer `main.py` està pensat per seer minimalista, amb la major part de la lògica i funcionalitats implementades en els altres fitxers del projecte, presents als directoris `models/` i `utils/`.

## Descripció dels directoris i fitxers principals

### Arrel del Projecte (`TFG/`)
- `main.py`: Fitxer principal per a l'execució de models i prediccions.
- `environment.yml`: Fitxer per a crear l'entorn de Conda amb tots els paquets necessaris.

### `data/` (dades a analitzar)
- `passatgers.csv`: Dataset principal.
- `hipoteques.csv`: Dataset secundari.
- `hipoteques_raw.csv`: Dataset brut, per netejar amb KNIME.

### `knime/` (workflow de KNIME)
- `TFG.knwf`: Workflow de KNIME emprat per a la preparació de les dades.

### `models/` (models i funcions associades)
- `arima.py`: Implementació del model ARIMA, fent ús de la classe `ARIMA` del mòdul `pmdarima`.
- `auto_arima.py`: Implementació del model ARIMA, fent ús del mètode `auto_arima` del mòdul `pmdarima`.
- `holt_winters.py`: Implementació del model de suavització exponencial triple, fent ús de la classe `ExponentialSmoothing` del paquet `statsmodels`.
- `prophet.py`: Implementació del model Prophet, fent ús del paquet `prophet`.

### `utils/` (scripts amb funcions auxiliars)
- `analysis.py`: Funcions per a analitzar les dades i efectuar comprovacions.
- `preprocessing.py`: Funcions per netejar i preparar les dades.
- `visualization.py`: Funcions per generar gràfiques.
- `utils`: Funcions complementàries.

### `tex/` (eixida del document)
- `main.pdf`: Document final.
