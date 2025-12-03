# src/gold_fact_sales.py
"""
Capa Gold - Master Table Unificada para ML
Carga tablas desde Silver y construye master table con todos los datos
"""
from prefect import flow, task, get_run_logger
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from typing import Dict
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from model_evaluation import evaluate_models_with_cv
import pickle
from typing import List
from business_kpis import calculate_business_kpis
from visualization import generate_all_visualizations
from visualization import generate_all_visualizations
from sla_analysis import analyze_sla_complete



from gold_features import (
    generate_temporal_features,
    generate_logistics_features,
    generate_payment_features,
    generate_customer_features,
    combine_features
)

# Cargar .env desde la raíz del proyecto
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

# Configurar carpeta de outputs (ruta relativa)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
PKLS_DIR = os.path.join(OUTPUT_DIR, 'pkls')
MODEL_METRICS_DIR = os.path.join(OUTPUT_DIR, 'model_metrics')
FEATURE_IMPORTANCES_DIR = os.path.join(OUTPUT_DIR, 'feature_importances')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PKLS_DIR, exist_ok=True)
os.makedirs(MODEL_METRICS_DIR, exist_ok=True)
os.makedirs(FEATURE_IMPORTANCES_DIR, exist_ok=True)

SILVER_CONN = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/silver"
GOLD_CONN   = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/gold"

engine_silver = create_engine(SILVER_CONN)
engine_gold   = create_engine(GOLD_CONN)


@task(log_prints=True)
def load_datasets_from_silver() -> Dict[str, pd.DataFrame]:
    """Carga todos los datasets desde Silver."""
    logger = get_run_logger()
    logger.info("📥 Cargando datasets desde Silver...")
    
    datasets = {}
    tables = {
        'customers': 'customers',
        'geolocation': 'geolocation',
        'order_items': 'order_items',
        'order_payments': 'order_payments',
        'order_reviews': 'order_reviews',
        'orders': 'orders',
        'products': 'products',
        'sellers': 'sellers',
        'category_translation': 'product_category_translation'
    }
    
    for key, table in tables.items():
        try:
            logger.info(f"   Cargando: {table}")
            df = pd.read_sql(f"SELECT * FROM curated.{table}", engine_silver)
            datasets[key] = df
            logger.info(f"   ✅ {key}: {len(df):,} registros")
        except Exception as e:
            logger.warning(f"   ⚠️  Error cargando {table}: {str(e)}")
            datasets[key] = pd.DataFrame()
    
    return datasets


@task(log_prints=True)
def build_master_table(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construye la master table unificada desde todas las fuentes.
    Replica la estructura de carga desde CSVs.
    """
    logger = get_run_logger()
    logger.info("🔨 Construyendo Master Table en Gold...")
    
    # Verificar datasets requeridos
    required = ['orders', 'order_items', 'customers', 'products', 'sellers']
    for table in required:
        if table not in datasets or datasets[table].empty:
            raise ValueError(f"Dataset requerido '{table}' no disponible")
    
    # Base: orders (filtrar solo delivered)
    master = datasets['orders'].copy()
    if 'order_status' in master.columns:
        master = master[master['order_status'] == 'delivered'].copy()
        logger.info(f"   Base (orders delivered): {len(master):,} registros")
    else:
        logger.info(f"   Base (orders): {len(master):,} registros")
    
    # Join con customers
    if not datasets['customers'].empty:
        master = master.merge(
            datasets['customers'],
            on='customer_id',
            how='left',
            suffixes=('', '_cust')
        )
        logger.info(f"   + customers: {len(master):,} registros")
    
    # Join con order_items
    if not datasets['order_items'].empty:
        master = master.merge(
            datasets['order_items'],
            on='order_id',
            how='left',
            suffixes=('', '_item')
        )
        logger.info(f"   + order_items: {len(master):,} registros")
    
    # Join con products
    if not datasets['products'].empty:
        master = master.merge(
            datasets['products'],
            on='product_id',
            how='left',
            suffixes=('', '_prod')
        )
        logger.info(f"   + products: {len(master):,} registros")
    
    # Join con sellers
    if not datasets['sellers'].empty:
        master = master.merge(
            datasets['sellers'],
            on='seller_id',
            how='left',
            suffixes=('', '_sell')
        )
        logger.info(f"   + sellers: {len(master):,} registros")
    
    # Join con order_payments
    if not datasets['order_payments'].empty:
        master = master.merge(
            datasets['order_payments'],
            on='order_id',
            how='left',
            suffixes=('', '_pay')
        )
        logger.info(f"   + order_payments: {len(master):,} registros")
    
    # Join con order_reviews
    if not datasets['order_reviews'].empty:
        master = master.merge(
            datasets['order_reviews'],
            on='order_id',
            how='left',
            suffixes=('', '_rev')
        )
        logger.info(f"   + order_reviews: {len(master):,} registros")
    
    # Join con category_translation
    if not datasets['category_translation'].empty and 'product_category_name' in master.columns:
        master = master.merge(
            datasets['category_translation'],
            on='product_category_name',
            how='left',
            suffixes=('', '_trans')
        )
        logger.info(f"   + category_translation: {len(master):,} registros")
    
    # Nota: geolocation se puede usar después para calcular distancias
    # pero no se hace join directo porque tiene múltiples registros por zip
    
        # Nota: geolocation se puede usar después para calcular distancias
    # pero no se hace join directo porque tiene múltiples registros por zip
    
    # ============================================================
    # ELIMINAR DUPLICADOS POR ORDER_ID
    # ============================================================
    logger.info("   🧹 Verificando duplicados...")
    initial_count = len(master)
    
    # Contar duplicados por order_id
    duplicates_count = master.duplicated(subset=['order_id'], keep='first').sum()
    
    if duplicates_count > 0:
        logger.warning(f"      ⚠️  Duplicados detectados: {duplicates_count:,} registros")
        logger.info(f"      • Causa: Órdenes con múltiples items/pagos/reviews")
        
        # Eliminar duplicados manteniendo el primer registro
        master = master.drop_duplicates(subset=['order_id'], keep='first')
        
        logger.info(f"      ✅ Duplicados eliminados: {initial_count - len(master):,}")
        logger.info(f"      📊 Registros únicos finales: {len(master):,}")
    else:
        logger.info(f"      ✅ No se detectaron duplicados en order_id")
    
    # ============================================================
    # CALCULAR TARGET: Delayed_time
    # ============================================================
    if 'order_delivered_customer_date' in master.columns and 'order_estimated_delivery_date' in master.columns:
        logger.info("   🎯 Calculando target 'Delayed_time'...")
        
        # Convertir a datetime
        master['order_delivered_customer_date'] = pd.to_datetime(master['order_delivered_customer_date'])
        master['order_estimated_delivery_date'] = pd.to_datetime(master['order_estimated_delivery_date'])
        
        # Calcular diferencia en días
        master['Delayed_time'] = (
            master['order_delivered_customer_date'] - 
            master['order_estimated_delivery_date']
        ).dt.days
        
        logger.info(f"      • Rango: [{master['Delayed_time'].min():.0f}, {master['Delayed_time'].max():.0f}] días")
        logger.info(f"      • Media: {master['Delayed_time'].mean():.1f} días")
        logger.info(f"      • Valores nulos: {master['Delayed_time'].isna().sum():,}")
    
    # ============================================================
    # FILTRAR OUTLIERS
    # ============================================================
    initial_count = len(master)
    
    if 'Delayed_time' in master.columns:
        logger.info("   🧹 Filtrando outliers en Delayed_time...")
        
        # Filtrar rango válido: -10 a +20 días
        master = master[
            (master['Delayed_time'] >= -5) & 
            (master['Delayed_time'] <= 20)
        ].copy()
        
        removed = initial_count - len(master)
        logger.info(f"      • Outliers removidos: {removed:,} registros ({removed/initial_count*100:.1f}%)")
        logger.info(f"      • Registros finales: {len(master):,}")
    
    logger.info(f"   ✅ Master Table: {len(master):,} registros, {len(master.columns)} columnas")
    
    logger.info(f"   ✅ Master Table: {len(master):,} registros, {len(master.columns)} columnas")
    
    return master

@task(log_prints=True)
def apply_one_hot_encoding(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding a las columnas categóricas de la master table.
    """
    logger = get_run_logger()
    logger.info("🔢 Aplicando One-Hot Encoding a columnas categóricas...")
    
    initial_columns = len(master_df.columns)
    
    # Definir columnas categóricas para One-Hot Encoding
    categorical_columns = []
    
    # 1. Order status (si existe y no fue filtrado)
    if 'order_status' in master_df.columns:
        categorical_columns.append('order_status')
    
    # 2. Product category
    if 'product_category_name' in master_df.columns:
        categorical_columns.append('product_category_name')
    
    # 3. Payment type
    if 'payment_type' in master_df.columns:
        categorical_columns.append('payment_type')
    
    # 4. Customer state
    if 'customer_state' in master_df.columns:
        categorical_columns.append('customer_state')
    
    # 5. Seller state
    if 'seller_state' in master_df.columns:
        categorical_columns.append('seller_state')
    
    # 6. Customer city (CUIDADO: puede tener muchas categorías)
    # Solo si tiene menos de 50 valores únicos
    if 'customer_city' in master_df.columns:
        unique_cities = master_df['customer_city'].nunique()
        if unique_cities <= 50:
            categorical_columns.append('customer_city')
        else:
            logger.info(f"   ⚠️  customer_city tiene {unique_cities} valores únicos, se omite del encoding")
    
    # 7. Seller city (CUIDADO: puede tener muchas categorías)
    if 'seller_city' in master_df.columns:
        unique_seller_cities = master_df['seller_city'].nunique()
        if unique_seller_cities <= 50:
            categorical_columns.append('seller_city')
        else:
            logger.info(f"   ⚠️  seller_city tiene {unique_seller_cities} valores únicos, se omite del encoding")
    
    # Filtrar solo las columnas que existen
    categorical_columns = [col for col in categorical_columns if col in master_df.columns]
    
    if not categorical_columns:
        logger.info("   ℹ️  No se encontraron columnas categóricas para encoding")
        return master_df
    
    logger.info(f"   📋 Columnas a encodear: {categorical_columns}")
    
    # Aplicar One-Hot Encoding
    for col in categorical_columns:
        # Contar valores únicos
        unique_values = master_df[col].nunique()
        logger.info(f"      • {col}: {unique_values} categorías únicas")
        
        # Crear dummies
        dummies = pd.get_dummies(
            master_df[col], 
            prefix=col,
            drop_first=True,  # Evitar multicolinealidad
            dtype=int
        )
        
        # Agregar al dataframe
        master_df = pd.concat([master_df, dummies], axis=1)
        
        # Eliminar columna original
        master_df = master_df.drop(columns=[col])
        
        logger.info(f"         ✅ Creadas {len(dummies.columns)} columnas dummy")
    
    final_columns = len(master_df.columns)
    new_columns = final_columns - initial_columns
    
    logger.info(f"   ✅ One-Hot Encoding completado")
    logger.info(f"      • Columnas iniciales: {initial_columns}")
    logger.info(f"      • Columnas finales: {final_columns}")
    logger.info(f"      • Nuevas columnas: {new_columns}")
    
    return master_df

@task(log_prints=True)
def feature_selection(master_df: pd.DataFrame, target_col: str = 'Delayed_time', 
                     correlation_threshold: float = 0.05,
                     top_n_features: int = 50) -> pd.DataFrame:
    """
    Reduce variables mediante Feature Selection basado en el target.
    
    Métodos utilizados:
    1. Correlación de Pearson con el target
    2. Mutual Information
    3. Feature Importance de Random Forest
    
    Args:
        master_df: DataFrame con todas las features
        target_col: Nombre de la columna target
        correlation_threshold: Umbral mínimo de correlación absoluta
        top_n_features: Número máximo de features a mantener
    
    Returns:
        DataFrame con features seleccionadas + target + IDs
    """
    logger = get_run_logger()
    logger.info("🎯 Aplicando Feature Selection...")
    
    if target_col not in master_df.columns:
        logger.warning(f"   ⚠️  Target '{target_col}' no encontrado, se omite feature selection")
        return master_df
    
    # Separar columnas
    id_columns = [col for col in master_df.columns if 'id' in col.lower()]
    date_columns = [col for col in master_df.columns if master_df[col].dtype == 'datetime64[ns]']
    
    # Columnas numéricas (excluyendo IDs, fechas y target)
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols 
                   if col != target_col 
                   and col not in id_columns]
    
    logger.info(f"   📊 Análisis inicial:")
    logger.info(f"      • Total columnas: {len(master_df.columns)}")
    logger.info(f"      • Columnas numéricas: {len(numeric_cols)}")
    logger.info(f"      • Features candidatas: {len(feature_cols)}")
    logger.info(f"      • IDs: {len(id_columns)}")
    logger.info(f"      • Fechas: {len(date_columns)}")
    
    if len(feature_cols) == 0:
        logger.warning("   ⚠️  No hay features numéricas para seleccionar")
        return master_df
    
    # Preparar datos (eliminar NaN)
    X = master_df[feature_cols].fillna(0)
    y = master_df[target_col].fillna(0)
    
    # ============================================================
    # MÉTODO 1: Correlación de Pearson
    # ============================================================
    logger.info("   📈 Método 1: Correlación de Pearson...")
    correlations = {}
    for col in feature_cols:
        corr = X[col].corr(y)
        correlations[col] = abs(corr)
    
    corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['correlation'])
    corr_df = corr_df.sort_values('correlation', ascending=False)
    
    # Features con correlación significativa
    significant_corr = corr_df[corr_df['correlation'] >= correlation_threshold].index.tolist()
    logger.info(f"      • Features con |corr| >= {correlation_threshold}: {len(significant_corr)}")
    
    # ============================================================
    # MÉTODO 2: Mutual Information
    # ============================================================
    logger.info("   🔗 Método 2: Mutual Information...")
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({'feature': feature_cols, 'mi_score': mi_scores})
    mi_df = mi_df.sort_values('mi_score', ascending=False)
    
    # Top features por MI
    top_mi = mi_df.head(top_n_features)['feature'].tolist()
    logger.info(f"      • Top {min(top_n_features, len(feature_cols))} features por MI")
    
    # ============================================================
    # MÉTODO 3: Random Forest Feature Importance
    # ============================================================
    logger.info("   🌲 Método 3: Random Forest Feature Importance...")
    
    # Entrenar RF rápido (pocos árboles para velocidad)
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    # Obtener importancias
    importances = rf.feature_importances_
    rf_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    rf_df = rf_df.sort_values('importance', ascending=False)
    
    # Top features por RF
    top_rf = rf_df.head(top_n_features)['feature'].tolist()
    logger.info(f"      • Top {min(top_n_features, len(feature_cols))} features por RF")
    
    # ============================================================
    # COMBINAR MÉTODOS
    # ============================================================
    logger.info("   🔀 Combinando métodos...")
    
    # Unión de features seleccionadas por los 3 métodos
    selected_features = list(set(significant_corr) | set(top_mi) | set(top_rf))
    
    # Limitar al top_n si es necesario
    if len(selected_features) > top_n_features:
        # Ordenar por promedio de rankings
        feature_scores = {}
        for feat in selected_features:
            corr_rank = list(corr_df.index).index(feat) if feat in corr_df.index else len(feature_cols)
            mi_rank = list(mi_df['feature']).index(feat) if feat in mi_df['feature'].values else len(feature_cols)
            rf_rank = list(rf_df['feature']).index(feat) if feat in rf_df['feature'].values else len(feature_cols)
            feature_scores[feat] = (corr_rank + mi_rank + rf_rank) / 3
        
        # Ordenar por score promedio
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1])
        selected_features = [f[0] for f in sorted_features[:top_n_features]]
    
    logger.info(f"   ✅ Features seleccionadas: {len(selected_features)}")
    
    # ============================================================
    # CREAR DATAFRAME REDUCIDO
    # ============================================================
    # Mantener: IDs + features seleccionadas + target + fechas importantes
    important_dates = ['order_purchase_timestamp', 'order_delivered_customer_date', 
                      'order_estimated_delivery_date']
    keep_dates = [col for col in important_dates if col in master_df.columns]
    
    final_columns = id_columns + selected_features + [target_col] + keep_dates
    final_columns = [col for col in final_columns if col in master_df.columns]
    
    reduced_df = master_df[final_columns].copy()
    
    # ============================================================
    # RESUMEN
    # ============================================================
    logger.info("")
    logger.info("   📊 Resumen de Feature Selection:")
    logger.info(f"      • Columnas originales: {len(master_df.columns)}")
    logger.info(f"      • Columnas finales: {len(reduced_df.columns)}")
    logger.info(f"      • Reducción: {len(master_df.columns) - len(reduced_df.columns)} columnas ({(1 - len(reduced_df.columns)/len(master_df.columns))*100:.1f}%)")
    logger.info("")
    logger.info("   🏆 Top 10 features por importancia combinada:")
    for i, feat in enumerate(selected_features[:10], 1):
        corr_val = correlations.get(feat, 0)
        logger.info(f"      {i}. {feat} (corr: {corr_val:.3f})")
    
    return reduced_df

@task(log_prints=True)
def optimize_xgboost_hyperparameters(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_val: pd.DataFrame, 
    y_val: pd.Series,
    n_trials: int = 50
) -> Dict:
    """
    Optimiza hiperparámetros de XGBoost usando Optuna.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        n_trials: Número de trials de Optuna
    
    Returns:
        Dict con mejores hiperparámetros
    """
    logger = get_run_logger()
    logger.info(f"🔍 Optimizando hiperparámetros con Optuna ({n_trials} trials)...")
    
    def objective(trial):
        """Función objetivo para Optuna."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),  # ✅ Más árboles
            'max_depth': trial.suggest_int('max_depth', 6, 20),  # ✅ Más profundidad
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.2, log=True),  # ✅ LR más bajo
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # ✅ Más variación
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # ✅ Ampliado
            'gamma': trial.suggest_float('gamma', 0.0, 2.0),  # ✅ Ampliado
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),  # ✅ Más regularización L1
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),  # ✅ Más regularización L2
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 5),  # ✅ NUEVO parámetro
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Entrenar modelo con estos hiperparámetros
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predecir en validación
        y_pred = model.predict(X_val)
        
        # Calcular R² (métrica a MAXIMIZAR)
        r2 = r2_score(y_val, y_pred)
        
        return -r2  # ✅ Negativo porque Optuna minimiza
    
    # Crear estudio de Optuna
    study = optuna.create_study(
        direction='minimize',  # Minimizar -R² (= maximizar R²)
        sampler=TPESampler(seed=42),
        study_name='xgboost_optimization'
    )
    
    # Ejecutar optimización
    study.optimize(
        objective, 
        n_trials=n_trials,
        show_progress_bar=False,
        n_jobs=1  # Usar 1 job porque XGBoost ya usa todos los cores
    )
    
    # Mejores hiperparámetros
    best_params = study.best_params
    best_mae = study.best_value
    
    logger.info("")
    logger.info("   ✅ Optimización completada")
    logger.info(f"      • Mejor MAE: {best_mae:.4f} días")
    logger.info(f"      • Trials completados: {len(study.trials)}")
    logger.info("")
    logger.info("   🏆 Mejores hiperparámetros:")
    for param, value in best_params.items():
        logger.info(f"      • {param}: {value}")
    
    return {
        'best_params': best_params,
        'best_mae': best_mae,
        'study': study
    }

@task(log_prints=True)
def train_xgboost_model(
    master_df: pd.DataFrame, 
    target_col: str = 'Delayed_time',
    use_optuna: bool = True,
    n_trials: int = 50
) -> Dict:
    """
    Entrena un modelo XGBoost Regressor para predecir el target.
    Opcionalmente usa Optuna para optimizar hiperparámetros.
    
    Args:
        master_df: DataFrame con features y target
        target_col: Nombre de la columna target
        use_optuna: Si True, usa Optuna para optimizar hiperparámetros
        n_trials: Número de trials de Optuna
    
    Returns:
        Dict con master_df actualizado y métricas del modelo
    """
    logger = get_run_logger()
    logger.info("🤖 Entrenando modelo XGBoost Regressor...")
    
    if target_col not in master_df.columns:
        logger.warning(f"   ⚠️  Target '{target_col}' no encontrado, se omite entrenamiento")
        return {'master_df': master_df, 'metrics': None}
    
    # ============================================================
    # PREPARAR DATOS
    # ============================================================
    logger.info("   📊 Preparando datos...")
    
    # Identificar columnas
    id_columns = [col for col in master_df.columns if 'id' in col.lower()]
    date_columns = [col for col in master_df.columns if master_df[col].dtype == 'datetime64[ns]']
    
    # Features numéricas (excluyendo IDs, fechas y target)
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols 
                   if col != target_col 
                   and col not in id_columns]
    
    logger.info(f"      • Features para entrenamiento: {len(feature_cols)}")
    logger.info(f"      • Registros totales: {len(master_df):,}")
    
    # Preparar X e y
    X = master_df[feature_cols].fillna(0)
    y = master_df[target_col].fillna(0)
    
    # Verificar que hay datos suficientes
    if len(X) < 100:
        logger.warning(f"   ⚠️  Datos insuficientes para entrenamiento ({len(X)} registros)")
        return {'master_df': master_df, 'metrics': None}
    
    # ============================================================
    # SPLIT TRAIN/VALIDATION/TEST
    # ============================================================
    logger.info("   ✂️  Dividiendo datos (60% train, 20% val, 20% test)...")
    
    # Primero: train+val (80%) vs test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Segundo: train (75% de 80% = 60%) vs val (25% de 80% = 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    
    logger.info(f"      • Train: {len(X_train):,} registros")
    logger.info(f"      • Validation: {len(X_val):,} registros")
    logger.info(f"      • Test: {len(X_test):,} registros")
    
    # ============================================================
    # OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA
    # ============================================================
    if use_optuna:
        optuna_result = optimize_xgboost_hyperparameters(
            X_train, y_train, 
            X_val, y_val,
            n_trials=n_trials
        )
        best_params = optuna_result['best_params']
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        best_params['verbosity'] = 0
    else:
        # Hiperparámetros por defecto
        logger.info("   ⚙️  Usando hiperparámetros por defecto...")
        best_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    # ============================================================
    # ENTRENAR MODELO FINAL
    # ============================================================
    logger.info("")
    logger.info("   🚀 Entrenando modelo final con mejores hiperparámetros...")
    
    model = xgb.XGBRegressor(**best_params)
    
    # Entrenar con train + validation
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    model.fit(
        X_train_full, y_train_full,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    logger.info(f"      ✅ Modelo entrenado")
    
    # ============================================================
    # EVALUAR MODELO
    # ============================================================
    logger.info("")
    logger.info("   📈 Evaluando modelo...")
    
    # Predicciones
    y_train_pred = model.predict(X_train_full)
    y_test_pred = model.predict(X_test)
    
    # Métricas en train
    train_mae = mean_absolute_error(y_train_full, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, y_train_pred))
    train_r2 = r2_score(y_train_full, y_train_pred)
    
    # Métricas en test
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    logger.info("")
    logger.info("   📊 Métricas del Modelo:")
    logger.info("      TRAIN:")
    logger.info(f"         • MAE:  {train_mae:.2f} días")
    logger.info(f"         • RMSE: {train_rmse:.2f} días")
    logger.info(f"         • R²:   {train_r2:.4f}")
    logger.info("      TEST:")
    logger.info(f"         • MAE:  {test_mae:.2f} días")
    logger.info(f"         • RMSE: {test_rmse:.2f} días")
    logger.info(f"         • R²:   {test_r2:.4f}")
    
    # ============================================================
    # GENERAR PREDICCIONES PARA TODO EL DATASET
    # ============================================================
    logger.info("")
    logger.info("   🔮 Generando predicciones para todo el dataset...")
    
    predictions = model.predict(X)
    master_df['Delayed_time_predicted'] = predictions
    
    # Calcular error de predicción
    master_df['prediction_error'] = master_df[target_col] - master_df['Delayed_time_predicted']
    master_df['prediction_error_abs'] = abs(master_df['prediction_error'])
    
    logger.info(f"      ✅ Predicciones agregadas como 'Delayed_time_predicted'")
    logger.info(f"      • Error promedio: {master_df['prediction_error_abs'].mean():.2f} días")
    logger.info(f"      • Error mediano: {master_df['prediction_error_abs'].median():.2f} días")
    
    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================
    logger.info("")
    logger.info("   🏆 Top 10 Features más importantes:")
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        logger.info(f"      {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # ============================================================
    # RETORNAR RESULTADOS
    # ============================================================
    metrics = {
        'train': {
            'mae': float(train_mae),
            'rmse': float(train_rmse),
            'r2': float(train_r2)
        },
        'test': {
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'r2': float(test_r2)
        },
        'n_features': len(feature_cols),
        'best_params': best_params if use_optuna else None
    }
    
    return {
        'master_df': master_df,
        'metrics': metrics,
        'model': model,
        'feature_cols': feature_cols
    }

@task(log_prints=True)
def save_model_pickle(
    model,
    model_metrics: Dict,
    feature_cols: List[str],
    output_path: str = None
) -> Dict:
    """
    Guarda el modelo entrenado en formato pickle.
    
    Args:
        model: Modelo entrenado
        model_metrics: Métricas del modelo
        feature_cols: Lista de features utilizadas
        output_path: Ruta donde guardar el modelo
    
    Returns:
        Dict con información del guardado
    """
    logger = get_run_logger()
    logger.info("💾 Guardando modelo en formato pickle...")
    
    # Definir ruta por defecto con timestamp
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(PKLS_DIR, f'xgboost_model_{timestamp}.pkl')
    
    # Crear diccionario con modelo y metadata
    model_package = {
        'model': model,
        'metrics': model_metrics,
        'feature_columns': feature_cols,
        'model_type': type(model).__name__,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Guardar en pickle
    with open(output_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    # Verificar tamaño del archivo
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    logger.info(f"   ✅ Modelo guardado exitosamente")
    logger.info(f"   📁 Ruta: {output_path}")
    logger.info(f"   📊 Tamaño: {file_size:.2f} MB")
    logger.info(f"   🎯 Features: {len(feature_cols)}")
    logger.info(f"   📈 Test MAE: {model_metrics['test']['mae']:.3f} días")
    logger.info(f"   📈 Test R²: {model_metrics['test']['r2']:.4f}")
    
    return {
        'output_path': output_path,
        'file_size_mb': file_size,
        'n_features': len(feature_cols)
    }

@task(log_prints=True)
def save_feature_importance(
    model,
    feature_cols: List[str],
    output_path: str = None
) -> str:
    """
    Guarda la importancia de features en CSV.
    
    Args:
        model: Modelo entrenado
        feature_cols: Lista de features
        output_path: Ruta del archivo de salida
    
    Returns:
        Ruta del archivo guardado
    """
    logger = get_run_logger()
    logger.info("💾 Guardando feature importance...")
    
    # Definir ruta por defecto con timestamp
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(FEATURE_IMPORTANCES_DIR, f'feature_importance_{timestamp}.csv')
    
    # Crear DataFrame con importancias
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Guardar
    feature_importance.to_csv(output_path, index=False)
    
    logger.info(f"   ✅ Feature importance guardado en: {output_path}")
    logger.info(f"   📊 Top 5 features:")
    for idx, row in feature_importance.head(5).iterrows():
        logger.info(f"      {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    return output_path

@task(log_prints=True)
def save_model_metrics(
    model_metrics: Dict,
    cv_results: Dict = None,
    output_path: str = None
) -> str:
    """
    Guarda las métricas del modelo en JSON.
    
    Args:
        model_metrics: Métricas del modelo final
        cv_results: Resultados de cross-validation
        output_path: Ruta del archivo de salida
    
    Returns:
        Ruta del archivo guardado
    """
    logger = get_run_logger()
    logger.info("💾 Guardando métricas del modelo...")
    
    # Definir ruta por defecto con timestamp
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(MODEL_METRICS_DIR, f'model_metrics_{timestamp}.json')
    
    # Preparar datos para guardar
    metrics_data = {
        'xgboost_final': model_metrics,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    if cv_results:
        metrics_data['cross_validation'] = {
            'best_model': cv_results['best_model'],
            'n_features': cv_results['n_features'],
            'n_samples': cv_results['n_samples']
        }
    
    # Guardar en JSON
    import json
    with open(output_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    logger.info(f"   ✅ Métricas guardadas en: {output_path}")
    
    return output_path

@task(log_prints=True)
def save_master_table(master_df: pd.DataFrame, datasets: Dict[str, pd.DataFrame]) -> Dict:
    """Guarda la master table y geolocation en Gold."""
    logger = get_run_logger()
    logger.info("💾 Guardando Master Table en Gold...")
    
    # Guardar master table
    master_df.to_sql(
        'master_table',
        engine_gold,
        schema='dm',
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=5000
    )
    
    logger.info(f"   ✅ Tabla 'gold.dm.master_table' creada")
    logger.info(f"   📊 {len(master_df):,} registros, {len(master_df.columns)} columnas")
    
    # Guardar geolocation por separado (para cálculos de distancia)
    if 'geolocation' in datasets and not datasets['geolocation'].empty:
        logger.info("   🗺️  Procesando geolocation (eliminando duplicados)...")
        geo_df = datasets['geolocation'].copy()
        
        # Convertir zip code a numérico
        geo_df['geolocation_zip_code_prefix'] = pd.to_numeric(
            geo_df['geolocation_zip_code_prefix'], 
            errors='coerce'
        )
        
        # Eliminar duplicados agrupando por zip code y promediando coordenadas
        geo_unique = geo_df.groupby('geolocation_zip_code_prefix').agg({
            'geolocation_lat': 'mean',
            'geolocation_lng': 'mean',
            'geolocation_city': 'first',
            'geolocation_state': 'first'
        }).reset_index()
        
        initial_count = len(geo_df)
        final_count = len(geo_unique)
        duplicates_removed = initial_count - final_count
        
        logger.info(f"      • Registros iniciales: {initial_count:,}")
        logger.info(f"      • Duplicados eliminados: {duplicates_removed:,}")
        logger.info(f"      • Registros únicos: {final_count:,}")
        
        # Guardar geolocation sin duplicados
        geo_unique.to_sql(
            'geolocation',
            engine_gold,
            schema='dm',
            if_exists='replace',
            index=False,
            method='multi',
            chunksize=5000
        )
        logger.info(f"   ✅ Tabla 'gold.dm.geolocation' creada ({final_count:,} registros únicos)")
    
    return {
        "status": "success",
        "total_rows": len(master_df),
        "total_columns": len(master_df.columns),
        "target_stats": {
            "min": float(master_df['Delayed_time'].min()) if 'Delayed_time' in master_df.columns else None,
            "max": float(master_df['Delayed_time'].max()) if 'Delayed_time' in master_df.columns else None,
            "mean": float(master_df['Delayed_time'].mean()) if 'Delayed_time' in master_df.columns else None
        }
    }


@flow(name="Silver → Gold (Master Table + Features)", log_prints=True)
def silver_to_gold():
    """
    Flujo que construye la master table y features completas en Gold
    desde todas las tablas de Silver
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("🥇 INICIANDO CONSTRUCCIÓN DE MASTER TABLE + FEATURES - GOLD")
    logger.info("=" * 80)
    
    # 1. Cargar datasets desde Silver
    datasets = load_datasets_from_silver()
    
    # 2. Construir master table base
    master_df = build_master_table(datasets)
    
    # 2.5. Aplicar One-Hot Encoding
    master_df = apply_one_hot_encoding(master_df)

    @task(log_prints=True)
    def remove_low_variance_features(master_df: pd.DataFrame, variance_threshold: float = 0.01) -> pd.DataFrame:
        """
        Elimina features con varianza muy baja (casi constantes).
        """
        logger = get_run_logger()
        logger.info(f"🧹 Eliminando features con varianza < {variance_threshold}...")
        
        numeric_cols = master_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols 
                    if col != 'Delayed_time' and 'id' not in col.lower()]
        
        low_variance_cols = []
        for col in feature_cols:
            if master_df[col].std() < variance_threshold:
                low_variance_cols.append(col)
        
        if low_variance_cols:
            logger.info(f"   • Features eliminadas: {len(low_variance_cols)}")
            master_df = master_df.drop(columns=low_variance_cols)
        else:
            logger.info(f"   • No se encontraron features con baja varianza")
        
        return master_df

    # Luego en el flujo, ANTES de feature_selection:
    master_df = remove_low_variance_features(master_df, variance_threshold=0.01)
    
    # 2.6. Feature Selection (reducir variables)
    master_df = feature_selection(
        master_df, 
        target_col='Delayed_time',
        correlation_threshold=0.005,
        top_n_features=80
    )

    logger.info(f"📊 Features seleccionadas: {[col for col in master_df.columns if col not in ['order_id', 'Delayed_time']]}")
    logger.info(f"📊 Total features numéricas: {len([col for col in master_df.select_dtypes(include=[np.number]).columns if col != 'Delayed_time'])}")

    # Después de las líneas de logging de features (línea 1047)
    logger.info("")
    logger.info("=" * 80)
    logger.info("🔍 DIAGNÓSTICO DETALLADO DE DATOS")
    logger.info("=" * 80)

    # 1. Información básica
    logger.info(f"📊 DATOS GENERALES:")
    logger.info(f"   • Registros totales: {len(master_df):,}")
    logger.info(f"   • Features numéricas: {len([col for col in master_df.select_dtypes(include=[np.number]).columns if col != 'Delayed_time'])}")

    # 2. Análisis del Target
    logger.info(f"")
    logger.info(f"🎯 ANÁLISIS DEL TARGET (Delayed_time):")
    logger.info(f"   • Media: {master_df['Delayed_time'].mean():.2f} días")
    logger.info(f"   • Mediana: {master_df['Delayed_time'].median():.2f} días")
    logger.info(f"   • Desviación estándar: {master_df['Delayed_time'].std():.2f} días")
    logger.info(f"   • Rango: [{master_df['Delayed_time'].min():.0f}, {master_df['Delayed_time'].max():.0f}] días")
    logger.info(f"   • Skewness: {master_df['Delayed_time'].skew():.2f}")
    logger.info(f"   • Valores nulos: {master_df['Delayed_time'].isna().sum()}")

    # 3. Distribución del target
    percentiles = master_df['Delayed_time'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])
    logger.info(f"   • Percentiles:")
    logger.info(f"      - 25%: {percentiles[0.25]:.1f} días")
    logger.info(f"      - 50%: {percentiles[0.50]:.1f} días")
    logger.info(f"      - 75%: {percentiles[0.75]:.1f} días")
    logger.info(f"      - 90%: {percentiles[0.90]:.1f} días")
    logger.info(f"      - 95%: {percentiles[0.95]:.1f} días")

    # 4. Varianza de las features
    logger.info(f"")
    logger.info(f"📈 ANÁLISIS DE FEATURES:")
    numeric_cols = [col for col in master_df.select_dtypes(include=[np.number]).columns 
                    if col != 'Delayed_time' and 'id' not in col.lower()]

    # Features con baja varianza (casi constantes)
    low_variance_features = []
    for col in numeric_cols:
        std = master_df[col].std()
        if std < 0.01:
            low_variance_features.append(col)

    if low_variance_features:
        logger.warning(f"   ⚠️  Features con baja varianza (<0.01): {len(low_variance_features)}")
        logger.warning(f"      {low_variance_features[:5]}")  # Mostrar primeras 5

    # Features con alta correlación con el target
    logger.info(f"")
    logger.info(f"🔗 TOP 10 FEATURES CON MAYOR CORRELACIÓN CON TARGET:")
    correlations = {}
    for col in numeric_cols:
        corr = abs(master_df[col].corr(master_df['Delayed_time']))
        if not np.isnan(corr):
            correlations[col] = corr

    top_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (feat, corr) in enumerate(top_corr, 1):
        logger.info(f"   {i}. {feat}: {corr:.4f}")

    # 5. Verificar valores nulos
    logger.info(f"")
    logger.info(f"❓ VALORES NULOS EN FEATURES:")
    null_counts = master_df[numeric_cols].isnull().sum()
    features_with_nulls = null_counts[null_counts > 0].sort_values(ascending=False)
    if len(features_with_nulls) > 0:
        logger.warning(f"   ⚠️  Features con valores nulos: {len(features_with_nulls)}")
        for feat, count in features_with_nulls.head(5).items():
            logger.warning(f"      • {feat}: {count:,} ({count/len(master_df)*100:.1f}%)")
    else:
        logger.info(f"   ✅ No hay valores nulos en las features")

    logger.info("=" * 80)
    logger.info("")


    # ============================================================
    # EVALUACIÓN CON VALIDACIÓN CRUZADA
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("🔬 EVALUACIÓN DE MODELOS CON VALIDACIÓN CRUZADA")
    logger.info("=" * 80)
    
    # Evaluar múltiples modelos con CV
    cv_results = evaluate_models_with_cv(
        master_df,
        target_col='Delayed_time',
        cv_folds=5,
        save_results=True
    )
    
    best_model_name = cv_results['best_model']
    logger.info(f"")
    logger.info(f"🏆 Mejor modelo identificado: {best_model_name}")
    logger.info(f"   • Test MAE promedio: {cv_results['results_df'].iloc[0]['test_mae_mean']:.3f} días")
    logger.info(f"   • Features utilizadas: {cv_results['n_features']}")
    logger.info(f"")
    
    # 2.7. Entrenar modelo XGBoost con Optuna y generar predicciones
    xgb_result = train_xgboost_model(
        master_df, 
        target_col='Delayed_time',
        use_optuna=True,  # Activar optimización con Optuna
        n_trials=50  # Número de trials (ajustable: 30-100)
    )
    master_df = xgb_result['master_df']
    model_metrics = xgb_result['metrics']

    # ============================================================
    # GUARDAR MODELO EN PICKLE
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("💾 GUARDANDO MODELO Y RESULTADOS")
    logger.info("=" * 80)
    
    # 1. Guardar modelo
    save_result = save_model_pickle(
        model=xgb_result['model'],
        model_metrics=model_metrics,
        feature_cols=xgb_result['feature_cols']
    )
    
    # 2. Guardar feature importance
    feature_importance_path = save_feature_importance(
        model=xgb_result['model'],
        feature_cols=xgb_result['feature_cols']
    )
    
    # 3. Guardar métricas completas
    metrics_path = save_model_metrics(
        model_metrics=model_metrics,
        cv_results=cv_results
    )
    
    # Resumen de archivos guardados
    logger.info("")
    logger.info("=" * 80)
    logger.info("📁 ARCHIVOS GUARDADOS EN OUTPUTS")
    logger.info("=" * 80)
    logger.info(f"   📦 Modelo: {save_result['output_path']}")
    logger.info(f"      • Tamaño: {save_result['file_size_mb']:.2f} MB")
    logger.info(f"   📊 Feature Importance: {feature_importance_path}")
    logger.info(f"   📈 Métricas: {metrics_path}")
    logger.info(f"   📋 Comparación CV: Ver carpeta outputs/")
    logger.info("=" * 80)

    # Resumen de archivos guardados
    logger.info("")
    logger.info("=" * 80)
    logger.info("📁 ARCHIVOS GUARDADOS EN OUTPUTS")
    logger.info("=" * 80)
    logger.info(f"   📦 Modelo: {save_result['output_path']}")
    logger.info(f"      • Tamaño: {save_result['file_size_mb']:.2f} MB")
    logger.info(f"   📊 Feature Importance: {feature_importance_path}")
    logger.info(f"   📈 Métricas: {metrics_path}")
    logger.info(f"   📋 Comparación CV: Ver carpeta outputs/")
    logger.info("=" * 80)
    
    # ============================================================
    # GENERAR VISUALIZACIONES
    # ============================================================
    viz_paths = generate_all_visualizations(
        master_df=master_df,
        model=xgb_result['model'],
        feature_cols=xgb_result['feature_cols'],
        model_metrics=model_metrics,
        cv_results=cv_results
    )

    # ============================================================
    # ANÁLISIS DE SLA
    # ============================================================
    sla_analysis_result = analyze_sla_complete(master_df)

    # ============================================================
    # CALCULAR KPIs DE NEGOCIO
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 ANÁLISIS DE KPIs DE NEGOCIO")
    logger.info("=" * 80)
    
    business_kpis = calculate_business_kpis(master_df)
    
    # 3. Guardar master_table en Gold (ahora con predicciones)
    result = save_master_table(master_df, datasets)
    
    # ============================================================
    # GENERAR FEATURES COMPLETAS
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎯 GENERANDO FEATURES COMPLETAS")
    logger.info("=" * 80)
    
    # 4. Generar features temporales
    logger.info("📅 Generando features temporales...")
    temporal_df = generate_temporal_features(master_df)
    
    # 5. Generar features logísticas
    logger.info("📦 Generando features logísticas...")
    logistics_df = generate_logistics_features(datasets)
    
    # 6. Generar features de pago
    logger.info("💳 Generando features de pago...")
    payment_df = generate_payment_features(datasets)
    
    # 7. Generar features de cliente
    logger.info("👥 Generando features de cliente...")
    customer_df = generate_customer_features(datasets)
    
    # 8. Combinar todas las features
    logger.info("🔗 Combinando todas las features...")
    features_df = combine_features(temporal_df, logistics_df, payment_df, customer_df)
    
    # 9. Guardar features completas en Gold
    logger.info("💾 Guardando features completas...")
    features_df.to_sql(
        'features',
        engine_gold,
        schema='dm',
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=5000
    )
    logger.info(f"   ✅ Tabla 'gold.dm.features' creada")
    logger.info(f"   📊 {len(features_df):,} registros, {len(features_df.columns)} columnas")
    
    # ============================================================
    # RESUMEN FINAL
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ GOLD LAYER COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"📊 Resumen:")
    logger.info(f"")
    logger.info(f"   MASTER TABLE:")
    logger.info(f"      • Registros: {result['total_rows']:,}")
    logger.info(f"      • Columnas: {result['total_columns']}")
    if result.get('target_stats'):
        logger.info(f"      • Target 'Delayed_time':")
        logger.info(f"         - Min: {result['target_stats']['min']:.0f} días")
        logger.info(f"         - Max: {result['target_stats']['max']:.0f} días")
        logger.info(f"         - Media: {result['target_stats']['mean']:.1f} días")
    
    # Agregar métricas del modelo
    if model_metrics:
        logger.info(f"")
        logger.info(f"   MODELO XGBOOST:")
        logger.info(f"      • Test MAE: {model_metrics['test']['mae']:.2f} días")
        logger.info(f"      • Test RMSE: {model_metrics['test']['rmse']:.2f} días")
        logger.info(f"      • Test R²: {model_metrics['test']['r2']:.4f}")
        logger.info(f"      • Features usadas: {model_metrics['n_features']}")
    logger.info(f"")
    logger.info(f"   FEATURES TABLE:")
    logger.info(f"      • Registros: {len(features_df):,}")
    logger.info(f"      • Columnas: {len(features_df.columns)}")
    logger.info(f"")
    logger.info(f"   GEOLOCATION:")
    logger.info(f"      • Tabla sin duplicados guardada")
    logger.info("=" * 80)
    
    # Actualizar resultado con info de features
    result['features'] = {
        'total_rows': len(features_df),
        'total_columns': len(features_df.columns)
    }
    
    return result


if __name__ == "__main__":
    silver_to_gold()