# Projecte TFG - Anàlisi Predictiva de Sèries Temporals

Aquest projecte explora algoritmes i models aplicats a l'anàlisi de sèries temporals, incloent-hi models com ARIMA, SARIMA, Holt-Winters, i altres tècniques de predicció. Aquest `README.md` proporciona una visió general de l'estructura del projecte, amb una explicació de cada directori i fitxer principal.

L'estructura del projecte està dissenyada per a ser modular i fàcil de mantindre, amb funcions i classes separades per a cada tasca específica. 
El fitxer `main.py` està pensat per seer minimalista, amb la major part de la lògica i funcionalitats implementades en els altres fitxers del projecte, presents als directoris `models/` i `utils/`.

## Descripció dels directoris i fitxers principals

### Arrel del Projecte (`TFG/`)
- `README.md`: Document d'instruccions i explicació general del projecte.
- `main.py`: Fitxer principal per a l'execució de models i prediccions.

### `data/` (dades a analitzar)
- `dataset3.csv`: Dataset principal per al projecte, amb la interpolació ja aplicada.

### `knime/` (workflow de KNIME)
- `TFG.knwf`: Workflow de KNIME emprat per a la preparació de les dades.

### `models/` (models i funcions associades)
- `arima.py`: Implementació de l'algoritme ARIMA, fent ús de la funció `auto_arima` del paquet `pmdarima`.

### `utils/` (scripts amb funcions auxiliars)
- `analysis.py`: Funcions per a analitzar les dades i efectuar comprovacions.
- `preprocessing.py`: Funcions per netejar i preparar les dades, lligades al dataset.
- `visualization.py`: Funcions per obtindre les diferents gràfiques requerides.

### `out/` (eixida del document)
- `main.pdf`: PDF amb el treball redactat, generat a partir dels fitxers `.tex`.

### `scripts/` (proves diverses)

### `tex/` (fitxers LaTeX i imatges)
- `main.tex`: Arxiu principal LaTeX que compila la memòria del projecte.
- `preambul.tex`: Configuració i estils globals de LaTeX.
- `capitols/`: Carpeta amb els capítols del projecte, des de la introducció fins a les conclusions.
- `imatges/`: Conté les imatges i logotips necessaris per a la documentació.
