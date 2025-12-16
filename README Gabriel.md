# Proyecto de Ciencia de Datos – Modelo de Riesgo de Crédito

Este proyecto desarrolla un **modelo de riesgo de crédito** cuyo objetivo es predecir la **probabilidad de impago** de un cliente a partir de información financiera, demográfica y de comportamiento crediticio.  
El enfoque del proyecto va más allá de la clasificación tradicional, incorporando **calibración de probabilidades** y **explicabilidad del modelo**, aspectos clave en entornos reales y regulados como el sector financiero.

---

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

- **`src/train_xgb.py`** → Script principal de entrenamiento, evaluación y calibración del modelo.
- **`src/features/`** → Módulos de ingeniería de variables y explicabilidad:
  - Agregaciones de tablas auxiliares
  - Generación de explicaciones SHAP globales y locales
- **`src/explore.ipynb`** → Notebook de exploración inicial del dataset.
- **`data/`** → Datasets en distintas fases:
  - **`data/raw/`** → Datos originales.
  - **`data/processed/`** → Dataset final con variables agregadas.
- **`reports/figures/`** → Gráficos generados:
  - Curva ROC
  - Matriz de confusión
  - Curva de calibración
  - SHAP summary plot
  - SHAP explicación individual
- **`requirements.txt`** → Dependencias del proyecto.

---

## Dataset

- **Fuente:** Home Credit Default Risk (Kaggle)
- **Número de registros:** 307.511 clientes
- **Número de variables:** 134 características tras ingeniería de variables
- **Variable objetivo:** `TARGET`
  - `0`: cliente sin impago
  - `1`: cliente con impago

El dataset contiene información relacionada con:
- Datos personales y laborales
- Situación financiera
- Historial de créditos
- Comportamiento de pagos

---

## Ingeniería de Variables

Se integraron varias tablas auxiliares a nivel de cliente, incluyendo:
- Historial de pagos de cuotas
- Créditos anteriores
- Balances de créditos POS

Las tablas fueron agregadas utilizando estadísticas como media, suma, mínimo y máximo, generando un único registro por cliente.

---

## Modelo

- **Algoritmo:** XGBoost Classifier
- **Tipo:** Gradient Boosted Trees
- **Manejo de variables categóricas:** habilitado (`enable_categorical=True`)
- **División de datos:** Train / Validation estratificado (80 / 20)

XGBoost fue seleccionado por su alto rendimiento en datos tabulares y su uso habitual en modelos de riesgo de crédito.

---

## Evaluación del Modelo

### Capacidad de Discriminación

- **Métrica:** ROC-AUC
- **Resultado:** **0.7677**

El modelo presenta una buena capacidad para diferenciar entre clientes con y sin riesgo de impago.

---

### Selección del Umbral de Decisión

Se evaluaron distintos umbrales de clasificación.  
El umbral seleccionado fue **0.2**, priorizando la **detección de clientes con riesgo de impago**, incluso a costa de aumentar falsos positivos.

Este enfoque es coherente con políticas conservadoras de gestión del riesgo.

---

### Matriz de Confusión

La matriz de confusión para el umbral seleccionado muestra un equilibrio razonable entre:
- Captura de clientes de riesgo
- Control del volumen de aprobaciones incorrectas

---

## Calibración de Probabilidades

Dado que los modelos basados en árboles suelen producir probabilidades descalibradas, se aplicó **calibración mediante regresión isotónica**.

- **Brier score (sin calibrar):** 0.0670  
- **Brier score (calibrado):** 0.0668  

La curva de calibración muestra una mejora clara, acercándose a la diagonal ideal, lo que permite interpretar las predicciones como **probabilidades reales de impago**.

---

## Interpretabilidad del Modelo (XAI)

### Explicabilidad Global

Se utilizó SHAP para analizar la importancia de las variables a nivel global.  
Las variables más relevantes incluyen:
- `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`
- Variables financieras y de comportamiento de pago

---

### Explicabilidad Local

Mediante gráficos SHAP individuales, se puede explicar cada predicción a nivel de cliente, mostrando cómo cada variable contribuye al riesgo estimado.

Este enfoque aporta transparencia y permite justificar decisiones de crédito.

---

## Ejecución del Proyecto

### Instalación de dependencias

```bash
pip install -r requirements.txt

### Instalación de dependencias

```bash
python src/train_xgb.py

Los resultados se guardan automáticamente en el directorio reports/figures/