# Machine Learning Pipeline - Predicción de Entregas Olist

Pipeline de Machine Learning para predecir retrasos en entregas utilizando datos de e-commerce de Olist.

## 📋 Tabla de Contenidos

- [Requisitos](#requisitos)
- [Estructura de Carpetas](#estructura-de-carpetas)
- [Instalación](#instalación)
- [Configuración de Datos](#configuración-de-datos)
- [Uso](#uso)
- [Descripción del Pipeline](#descripción-del-pipeline)
- [Modelos Implementados](#modelos-implementados)
- [Outputs](#outputs)

## 🔧 Requisitos

- **Python 3.12** (requerido)
- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)

## 📁 Estructura de Carpetas

Antes de ejecutar el proyecto, asegúrate de crear la siguiente estructura de carpetas:

```
code/
├── data/
│   ├── raw/                    # Datos originales (CSV de Olist)
│   ├── processed/              # Datos procesados (generado automáticamente)
│   └── features/               # Features generados (generado automáticamente)
├── models/                     # Modelos entrenados guardados (.pkl)
├── outputs/                    # Visualizaciones y resultados
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── clean.py           # Funciones de limpieza de datos
│   ├── features/
│   │   ├── __init__.py
│   │   └── make_features.py   # Generación de features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py           # Entrenamiento de modelos
│   │   ├── evaluate.py        # Evaluación de modelos
│   │   └── visualize.py       # Visualizaciones
│   └── pipeline.py            # Pipeline principal
├── requirements.txt
├── .gitignore
└── README.md
```

### Crear Carpetas Necesarias

Ejecuta los siguientes comandos para crear las carpetas requeridas:

```bash
mkdir -p data/raw data/processed data/features
mkdir -p models
mkdir -p outputs
```

## 🚀 Instalación

### 1. Verificar Versión de Python

Asegúrate de tener Python 3.12 instalado:

```bash
python3.12 --version
```

Si no tienes Python 3.12, descárgalo desde [python.org](https://www.python.org/downloads/).

### 2. Crear Entorno Virtual

Es altamente recomendado usar un entorno virtual:

```bash
# Crear entorno virtual con Python 3.12
python3.12 -m venv .venv

# Activar entorno virtual
# En macOS/Linux:
source .venv/bin/activate

# En Windows:
.venv\Scripts\activate
```

### 3. Instalar Dependencias

Con el entorno virtual activado, instala las dependencias desde `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Dependencias Principales

El archivo `requirements.txt` incluye:

- **pandas** (2.3.3): Manipulación de datos
- **numpy** (2.3.5): Operaciones numéricas
- **scikit-learn** (1.7.2): Modelos de ML
- **xgboost** (3.1.1): Gradient Boosting
- **matplotlib** (3.10.7): Visualizaciones
- **seaborn** (0.13.2): Visualizaciones estadísticas
- **joblib** (1.5.2): Serialización de modelos

## 📊 Configuración de Datos

### Descargar Datasets de Olist

1. Descarga los datasets de Olist desde [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

2. Coloca los siguientes archivos CSV en la carpeta `data/raw/`:

   - `olist_customers_dataset.csv`
   - `olist_geolocation_dataset.csv`
   - `olist_order_items_dataset.csv`
   - `olist_order_payments_dataset.csv`
   - `olist_order_reviews_dataset.csv`
   - `olist_orders_dataset.csv` (o variantes como `olist_orders_before_3_months.csv`)
   - `olist_products_dataset.csv`
   - `olist_sellers_dataset.csv`
   - `product_category_name_translation.csv`

### Verificar Datos

Asegúrate de que todos los archivos CSV estén en `data/raw/`:

```bash
ls -la data/raw/
```

## 🎯 Uso

### Ejecutar Pipeline Completo

Una vez instaladas las dependencias y configurados los datos, ejecuta el pipeline:

```bash
# Asegúrate de estar en el directorio raíz del proyecto
cd ~/olist

# Activar entorno virtual (si no está activado)
source .venv/bin/activate

# Ejecutar pipeline
python3.12 src/pipeline.py
```

### Personalizar Ejecución

Puedes modificar el archivo `src/pipeline.py` para cambiar parámetros:

```python
if __name__ == "__main__":
    # Cambiar el archivo de órdenes a usar
    results = run_pipeline('olist_orders_before_3_months.csv')
    
    # O usar el dataset completo
    # results = run_pipeline('olist_orders_dataset.csv')
```

## 🔄 Descripción del Pipeline

El pipeline ejecuta los siguientes pasos automáticamente:

### 1. **Carga de Datos**
   - Lee todos los CSV desde `data/raw/`
   - Carga 9 datasets diferentes

### 2. **Limpieza de Datos**
   - Limpia valores nulos y duplicados
   - Convierte tipos de datos
   - Normaliza formatos de fecha

### 3. **Generación de Features**
   - Crea features temporales (día de semana, mes, hora)
   - Calcula features geográficas (distancias)
   - Genera features de productos y pagos
   - Crea variables objetivo:
     - `is_late_delivery`: Clasificación binaria (entrega tardía o no)
     - `delay_days`: Regresión (días de retraso)

### 4. **Entrenamiento - Clasificación**
   - Modelo: XGBoost Classifier
   - Predice si una entrega llegará tarde
   - Métricas: ROC AUC, F1 Score, Precision, Recall

### 5. **Entrenamiento - Regresión**
   - Modelos comparados:
     - Linear Regression
     - Random Forest
     - XGBoost
   - Predice cuántos días de retraso tendrá una entrega
   - Métricas: RMSE, MAE, R²

### 6. **Evaluación y Visualización**
   - Genera gráficos de análisis
   - Compara modelos
   - Analiza residuos

### 7. **Guardado de Resultados**
   - Modelos entrenados en `models/`
   - Visualizaciones en `outputs/`
   - Métricas en `outputs/regression_results.csv`

## 🤖 Modelos Implementados

### Clasificación
- **XGBoost Classifier**: Predice entregas tardías con alta precisión

### Regresión
- **Linear Regression**: Modelo baseline
- **Random Forest Regressor**: Modelo ensemble robusto
- **XGBoost Regressor**: Modelo de gradient boosting (mejor performance)

## 📈 Outputs

Después de ejecutar el pipeline, encontrarás:

### En `outputs/`:
- `regression_analysis.png`: Análisis de predicciones vs valores reales
- `model_comparison.png`: Comparación de métricas entre modelos
- `predictions_comparison.png`: Comparación visual de predicciones
- `regression_results.csv`: Tabla con todas las métricas

### En `models/`:
- `olist_orders_before_3_months_classification.pkl`: Modelo de clasificación guardado

### En Consola:
- Métricas detalladas de clasificación y regresión
- Análisis de residuos
- Resumen de performance

## 🐛 Troubleshooting

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Error: "FileNotFoundError: data/raw/..."
Verifica que todos los archivos CSV estén en `data/raw/`

### Error: Versión de Python incorrecta
```bash
# Verificar versión
python3.12 --version

# Recrear entorno virtual con Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Error: "ModuleNotFoundError: No module named 'src'"
Asegúrate de ejecutar el script desde el directorio raíz del proyecto:
```bash
cd ~/olist
python3.12 src/pipeline.py
```

## 📝 Notas Adicionales

- El pipeline utiliza `train_test_split` con 80/20 para entrenamiento/prueba
- Los warnings de sklearn y xgboost están suprimidos para una salida más limpia
- Los modelos se guardan en formato `.pkl` usando `joblib`
- Las visualizaciones se generan automáticamente en formato PNG

## 🔒 Archivos Ignorados (.gitignore)

Los siguientes archivos/carpetas están excluidos del control de versiones:
- `data/raw/*` (datasets originales)
- `data/processed/*` (datos procesados)
- `*.pkl` (modelos guardados)
- `*.csv` (archivos de datos)
- `outputs/*` (visualizaciones)
- `.venv/*` (entorno virtual)
- `__pycache__/` (archivos compilados de Python)

## 📧 Soporte

Para preguntas o problemas, consulta la documentación de cada módulo en el código fuente.

---

**Versión**: v2.2.0  
**Python**: 3.12  
**Última actualización**: Noviembre 2025