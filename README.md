# 🛍️ Olist E-Commerce - Predicción de Retrasos en Entregas

## 📋 Descripción del Proyecto

Sistema de análisis y predicción de retrasos en entregas para Olist, el marketplace brasileño más grande. Este proyecto implementa un pipeline completo de ETL y Machine Learning para predecir el tiempo de retraso (`Delayed_time`) en las entregas de pedidos.

### 🎯 Objetivo

Predecir el tiempo de retraso en días entre la fecha estimada de entrega y la fecha real de entrega, utilizando múltiples algoritmos de Machine Learning con validación cruzada.

### 🏗️ Arquitectura del Proyecto

```
Bronze (Raw Data) → Silver (Curated) → Gold (Analytics & ML)
```

- **Bronze**: Datos crudos desde archivos CSV
- **Silver**: Datos limpios y normalizados
- **Gold**: Master table con features engineered + Modelos ML

---

## 📁 Estructura del Proyecto

```
olist/
├── data/                           # Datos CSV originales
│   ├── olist_customers_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_products_dataset.csv
│   └── ...
├── src/                            # Código fuente
│   ├── bronze_to_raw.py           # ETL: CSV → PostgreSQL (Bronze)
│   ├── silver_curated.py          # ETL: Bronze → Silver (Limpieza)
│   ├── gold_fact_sales.py         # ETL: Silver → Gold (ML Pipeline)
│   ├── gold_features.py           # Feature Engineering
│   ├── model_evaluation.py        # Evaluación con Cross-Validation
│   ├── flow.py                    # Orquestación Prefect
│   └── conn.py                    # Configuración de conexiones
├── requirements.txt               # Dependencias Python
├── docker-compose.yml             # Configuración Docker
├── .env                          # Variables de entorno
└── README.md                     # Este archivo
```

---

## 🚀 Instalación y Configuración

### Prerrequisitos

- Docker y Docker Compose
- Python 3.9+
- PostgreSQL (via Docker)

### Paso 1: Clonar el Repositorio

```bash
git clone <repository-url>
cd olist
```

### Paso 2: Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres
POSTGRES_HOST=db
POSTGRES_PORT=5432
```

### Paso 3: Levantar Servicios con Docker

```bash
docker-compose up -d
```

Esto iniciará:
- PostgreSQL (puerto 5432)
- Prefect Server (puerto 4200)
- Jupyter Notebook (opcional, puerto 8888)

### Paso 4: Instalar Dependencias Python

```bash
pip install -r requirements.txt
```

**Dependencias principales:**
- `prefect==2.18.1` - Orquestación de workflows
- `pandas==2.2.2` - Manipulación de datos
- `scikit-learn==1.7.2` - Machine Learning
- `xgboost==3.1.1` - Gradient Boosting
- `optuna==3.5.0` - Optimización de hiperparámetros
- `sqlalchemy==2.0.35` - ORM para PostgreSQL
- `psycopg2-binary==2.9.9` - Driver PostgreSQL

---

## 🔄 Pipeline de Ejecución

### Flujo Completo

```bash
# Ejecutar pipeline completo (Bronze → Silver → Gold)
cd src
python flow.py
```

### Ejecución por Capas

#### 1️⃣ Bronze Layer (CSV → PostgreSQL)

```bash
python bronze_to_raw.py
```

**Qué hace:**
- Carga datos desde archivos CSV
- Crea base de datos `bronze` en PostgreSQL
- Guarda datos crudos en esquema `raw`

**Tablas creadas:**
- `bronze.raw.customers`
- `bronze.raw.orders`
- `bronze.raw.order_items`
- `bronze.raw.products`
- `bronze.raw.sellers`
- `bronze.raw.order_payments`
- `bronze.raw.order_reviews`
- `bronze.raw.geolocation`
- `bronze.raw.product_category_translation`

#### 2️⃣ Silver Layer (Limpieza y Normalización)

```bash
python silver_curated.py
```

**Qué hace:**
- Lee datos desde Bronze
- Limpia valores nulos y duplicados
- Normaliza tipos de datos
- Valida integridad referencial
- Guarda en base de datos `silver` esquema `curated`

**Transformaciones:**
- Conversión de fechas a formato datetime
- Normalización de códigos postales
- Limpieza de valores nulos
- Validación de foreign keys

#### 3️⃣ Gold Layer (Feature Engineering + ML)

```bash
python gold_fact_sales.py
```

**Qué hace:**

1. **Construcción de Master Table**
   - Join de todas las tablas relacionadas
   - Filtrado de órdenes entregadas (`order_status = 'delivered'`)
   - Cálculo del target: `Delayed_time = order_delivered_customer_date - order_estimated_delivery_date`
   - Eliminación de duplicados y outliers

2. **One-Hot Encoding**
   - Codificación de variables categóricas:
     - `product_category_name`
     - `payment_type`
     - `customer_state`
     - `seller_state`

3. **Feature Selection**
   - **Método 1**: Correlación de Pearson
   - **Método 2**: Mutual Information
   - **Método 3**: Random Forest Feature Importance
   - Selección de top 50 features más relevantes

4. **🆕 Evaluación con Validación Cruzada**
   - Comparación de 6 modelos:
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
     - Random Forest
     - Gradient Boosting
     - XGBoost
   - K-Fold Cross-Validation (5 folds)
   - Métricas: MAE, RMSE, R²
   - Análisis de overfitting
   - Ranking general de modelos

5. **Entrenamiento del Modelo Final (XGBoost)**
   - Optimización de hiperparámetros con Optuna (50 trials)
   - Split: 60% train, 20% validation, 20% test
   - Métricas en train y test
   - Feature importance

6. **💾 Exportación del Modelo**
   - Guardado en formato pickle (`.pkl`)
   - Incluye: modelo + métricas + features + metadata
   - Ubicación: `/workspace/xgboost_model_final.pkl`

7. **Feature Engineering Avanzado**
   - Features temporales (día, mes, año, día de semana)
   - Features logísticas (distancias, tiempos de envío)
   - Features de pago (métodos, cuotas, valores)
   - Features de cliente (recurrencia, comportamiento)

**Tablas creadas en Gold:**
- `gold.dm.master_table` - Tabla principal con predicciones
- `gold.dm.features` - Features completas para análisis
- `gold.dm.geolocation` - Geolocalización sin duplicados

---

## 📊 Evaluación de Modelos

### Modelos Comparados

| Modelo | Descripción | Uso |
|--------|-------------|-----|
| **Linear Regression** | Regresión lineal simple | Baseline |
| **Ridge Regression** | Regresión con regularización L2 | Control de overfitting |
| **Lasso Regression** | Regresión con regularización L1 | Feature selection |
| **Random Forest** | Ensemble de árboles de decisión | Robusto a outliers |
| **Gradient Boosting** | Boosting secuencial | Alta precisión |
| **XGBoost** | Gradient boosting optimizado | Mejor performance |

### Métricas de Evaluación

- **MAE (Mean Absolute Error)**: Error promedio en días
- **RMSE (Root Mean Squared Error)**: Penaliza errores grandes
- **R² (Coefficient of Determination)**: Varianza explicada (0-1)
- **Training Time**: Tiempo de entrenamiento

### Validación Cruzada

- **Método**: K-Fold Cross-Validation (5 folds)
- **Ventajas**:
  - Reduce overfitting
  - Estimación más robusta del rendimiento
  - Utiliza todos los datos para train y test

### Resultados Exportados

Los resultados se guardan en:
```
/workspace/model_comparison_results.csv
```

Columnas:
- `model_name`: Nombre del modelo
- `test_mae_mean`, `test_mae_std`: MAE en test ± desviación
- `test_rmse_mean`, `test_rmse_std`: RMSE en test ± desviación
- `test_r2_mean`, `test_r2_std`: R² en test ± desviación
- `train_mae_mean`, `train_r2_mean`: Métricas en train
- `fit_time_mean`: Tiempo de entrenamiento
- `overall_score`: Score general (0-1)

---

## 🤖 Uso del Modelo Guardado

### Cargar el Modelo

```python
import pickle
import pandas as pd

# Cargar modelo
with open('/workspace/xgboost_model_final.pkl', 'rb') as f:
    model_package = pickle.load(f)

# Extraer componentes
model = model_package['model']
features = model_package['feature_columns']
metrics = model_package['metrics']

print(f"Modelo: {model_package['model_type']}")
print(f"Features: {len(features)}")
print(f"Test MAE: {metrics['test']['mae']:.3f} días")
print(f"Test R²: {metrics['test']['r2']:.4f}")
```

### Hacer Predicciones

```python
# Cargar nuevos datos
new_data = pd.read_csv('new_orders.csv')

# Asegurarse de tener las mismas features
X_new = new_data[features].fillna(0)

# Predecir
predictions = model.predict(X_new)

# Interpretar
new_data['predicted_delay'] = predictions
print(new_data[['order_id', 'predicted_delay']].head())
```

### Ejemplo Completo

```python
def predict_delivery_delay(order_data: pd.DataFrame) -> pd.DataFrame:
    """
    Predice el retraso en días para nuevos pedidos.
    
    Args:
        order_data: DataFrame con datos del pedido
    
    Returns:
        DataFrame con predicciones
    """
    # Cargar modelo
    with open('/workspace/xgboost_model_final.pkl', 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    features = model_package['feature_columns']
    
    # Preparar datos
    X = order_data[features].fillna(0)
    
    # Predecir
    predictions = model.predict(X)
    
    # Agregar al DataFrame
    order_data['predicted_delay_days'] = predictions
    order_data['delay_category'] = pd.cut(
        predictions,
        bins=[-float('inf'), -5, 0, 5, float('inf')],
        labels=['Muy Adelantado', 'A Tiempo', 'Leve Retraso', 'Retraso Grave']
    )
    
    return order_data

# Uso
new_orders = pd.read_csv('new_orders.csv')
results = predict_delivery_delay(new_orders)
print(results[['order_id', 'predicted_delay_days', 'delay_category']])
```

---

## 📈 Monitoreo y Logs

### Prefect UI

Accede a la interfaz de Prefect para monitorear ejecuciones:

```bash
# Iniciar Prefect Server
prefect server start

# Abrir en navegador
http://localhost:4200
```

### Logs del Pipeline

Los logs se imprimen en consola con formato detallado:

```
🥇 INICIANDO CONSTRUCCIÓN DE MASTER TABLE + EVALUACIÓN DE MODELOS
================================================================================
📥 Cargando datasets desde Silver...
   ✅ customers: 99,441 registros
   ✅ orders: 99,441 registros
...
🏆 COMPARACIÓN DE MODELOS CON VALIDACIÓN CRUZADA
================================================================================
   🔄 Evaluando Linear Regression con 5-Fold CV...
      ✅ Test MAE: 8.234 ± 0.123
      ✅ Test RMSE: 10.567 ± 0.234
      ✅ Test R²: 0.4567 ± 0.0234
...
🏆 Mejor modelo identificado: XGBoost
   - Test MAE promedio: 7.123 días
   - Features utilizadas: 50
```

---

## 🔧 Configuración Avanzada

### Ajustar Hiperparámetros de Optuna

En [gold_fact_sales.py](cci:7://file:///Users/howard/Downloads/modulo13/sprint3/olist/src/gold_fact_sales.py:0:0-0:0), línea ~960:

```python
xgb_result = train_xgboost_model(
    master_df, 
    target_col='Delayed_time',
    use_optuna=True,
    n_trials=100  # Aumentar para mejor optimización (más lento)
)
```

### Cambiar Número de Folds en CV

En [gold_fact_sales.py](cci:7://file:///Users/howard/Downloads/modulo13/sprint3/olist/src/gold_fact_sales.py:0:0-0:0), línea ~890:

```python
cv_results = evaluate_models_with_cv(
    master_df,
    target_col='Delayed_time',
    cv_folds=10,  # Aumentar para validación más robusta
    save_results=True
)
```

### Modificar Feature Selection

En [gold_fact_sales.py](cci:7://file:///Users/howard/Downloads/modulo13/sprint3/olist/src/gold_fact_sales.py:0:0-0:0), línea ~950:

```python
master_df = feature_selection(
    master_df, 
    target_col='Delayed_time',
    correlation_threshold=0.03,  # Más bajo = más features
    top_n_features=100  # Aumentar para más features
)
```

---

## 📊 Análisis de Resultados

### Consultas SQL Útiles

```sql
-- Ver estadísticas de retrasos
SELECT 
    AVG(Delayed_time) as avg_delay,
    STDDEV(Delayed_time) as std_delay,
    MIN(Delayed_time) as min_delay,
    MAX(Delayed_time) as max_delay
FROM gold.dm.master_table;

-- Comparar predicción vs real
SELECT 
    order_id,
    Delayed_time as real_delay,
    Delayed_time_predicted as predicted_delay,
    prediction_error_abs as error
FROM gold.dm.master_table
ORDER BY prediction_error_abs DESC
LIMIT 10;

-- Análisis por categoría de producto
SELECT 
    product_category_name,
    COUNT(*) as orders,
    AVG(Delayed_time) as avg_delay,
    AVG(Delayed_time_predicted) as avg_predicted
FROM gold.dm.master_table
GROUP BY product_category_name
ORDER BY avg_delay DESC;
```

### Visualizaciones Recomendadas

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
df = pd.read_sql("SELECT * FROM gold.dm.master_table", engine_gold)

# 1. Distribución de retrasos
plt.figure(figsize=(10, 6))
sns.histplot(df['Delayed_time'], bins=50, kde=True)
plt.title('Distribución de Retrasos en Entregas')
plt.xlabel('Días de Retraso')
plt.ylabel('Frecuencia')
plt.show()

# 2. Real vs Predicho
plt.figure(figsize=(10, 6))
plt.scatter(df['Delayed_time'], df['Delayed_time_predicted'], alpha=0.5)
plt.plot([df['Delayed_time'].min(), df['Delayed_time'].max()], 
         [df['Delayed_time'].min(), df['Delayed_time'].max()], 
         'r--', lw=2)
plt.xlabel('Retraso Real (días)')
plt.ylabel('Retraso Predicho (días)')
plt.title('Predicción vs Realidad')
plt.show()

# 3. Feature Importance
feature_importance = pd.read_csv('/workspace/feature_importance.csv')
top_10 = feature_importance.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_10, x='importance', y='feature')
plt.title('Top 10 Features Más Importantes')
plt.xlabel('Importancia')
plt.show()
```

---

## 🐛 Troubleshooting

### Error: "No module named 'model_evaluation'"

```bash
# Asegúrate de estar en el directorio correcto
cd /Users/howard/Downloads/modulo13/sprint3/olist/src
python gold_fact_sales.py
```

### Error: "Connection refused" (PostgreSQL)

```bash
# Verificar que Docker esté corriendo
docker-compose ps

# Reiniciar servicios
docker-compose restart db
```

### Error: "Out of memory" durante entrenamiento

Reducir el número de trials de Optuna:

```python
n_trials=20  # En lugar de 50
```

### Modelo muy lento

- Reducir `n_estimators` en los modelos
- Reducir `cv_folds` de 5 a 3
- Usar menos features en [feature_selection](cci:1://file:///Users/howard/Downloads/modulo13/sprint3/olist/src/gold_fact_sales.py:328:0-483:21)

---

## 📝 Notas Importantes

### Target Variable

- **Delayed_time**: Diferencia en días entre entrega real y estimada
- **Rango válido**: -30 a +60 días (outliers filtrados)
- **Interpretación**:
  - Negativo: Entrega adelantada
  - Cero: Entrega a tiempo
  - Positivo: Entrega retrasada

### Consideraciones de Producción

1. **Reentrenamiento**: Reentrenar modelo mensualmente con datos nuevos
2. **Monitoreo**: Trackear MAE en producción vs entrenamiento
3. **Data Drift**: Verificar distribución de features periódicamente
4. **Versionado**: Guardar modelos con timestamp en el nombre

---

## 👥 Contribuidores

- **Autor**: Howard
- **Proyecto**: Olist E-Commerce Analytics
- **Fecha**: Noviembre 2025

---

## 📄 Licencia

Este proyecto es de uso educativo y de investigación.

---

## 🔗 Referencias

- [Olist Dataset - Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Prefect Documentation](https://docs.prefect.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

---

## 📞 Soporte

Para preguntas o issues, contactar al equipo de desarrollo.

**¡Happy Modeling! 🚀**