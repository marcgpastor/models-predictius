# Anàlisi Predictiva de Sèries Temporals

Aquest projecte analitza diferents algoritmes i models per a la predicció de sèries temporals, incloent **SARIMA, Holt-Winters i Prophet**.

L'estructura està dissenyada per ser modular i fàcil de mantindre, separant les funcionalitats en mòduls específics.  
El fitxer `main.py` actua com a punt d'entrada, delegant la major part de la lògica als mòduls dins dels directoris `models/` i `utils/`.

---

## 🚀 **Requisits i Instal·lació**

### 🔧 **Entorn Conda**
Per instal·lar totes les dependències recomanades:

```bash
conda env create -f environment.yml
conda activate prediccions
```
---

## 📁 **Estructura del projecte**

### 🌍 **Arrel del Projecte (`TFG/`)**
- **`main.py`** → Punt d'entrada per a l'execució de models i generació de prediccions.
- **`environment.yml`** → Definició de l'entorn Conda amb tots els paquets necessaris.

### 📊 **Dades (`data/`)**
- **`passatgers.csv`** → Dataset principal.
- **`hipoteques.csv`** → Dataset secundari.
- **`hipoteques_raw.csv`** → Dades en brut (a netejar amb KNIME).

### ⚙️ **Preprocessament amb KNIME (`knime/`)**
- **`TFG.knwf`** → Workflow de KNIME per a la preparació de les dades.

### 🔬 **Models predictius (`models/`)**
- **`arima.py`** → Implementació del model ARIMA utilitzant `pmdarima.ARIMA`.
- **`auto_arima.py`** → Implementació d'ARIMA amb selecció automàtica de paràmetres (`pmdarima.auto_arima`).
- **`holt_winters.py`** → Implementació del model Holt-Winters (`statsmodels.ExponentialSmoothing`).
- **`prophet.py`** → Implementació del model Prophet (`prophet`).

### 💾 **Models guardats (`saved_models/`)**
- Fitxers amb els models preentrenats.

### 🛠 **Utilitats (`utils/`)**
- **`analysis.py`** → Funcions per a l'anàlisi i validació de dades.
- **`preprocessing.py`** → Funcions per a la neteja i preparació de dades.
- **`visualization.py`** → Funcions per a la generació de gràfiques.
- **`utils.py`** → Funcions auxiliars diverses.

---

## 🏁 **Ús del projecte**
Executa `main.py` per a entrenar un model i fer prediccions:

```bash
python main.py
```

---

## 🔖 **Autoria**
Aquest projecte ha estat desenvolupat per **Marc González Pastor**.  
Llicenciat sota les condicions establertes en el fitxer [`LICENSE`](LICENSE).
