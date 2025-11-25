# src/gold_fact_sales.py
"""
Capa Gold - Master Table Unificada para ML
Carga tablas desde Silver y construye master table con todos los datos
"""
from prefect import flow, task, get_run_logger
from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Dict

load_dotenv("/workspace/.env")

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
            (master['Delayed_time'] >= -30) & 
            (master['Delayed_time'] <= 60)
        ].copy()
        
        removed = initial_count - len(master)
        logger.info(f"      • Outliers removidos: {removed:,} registros ({removed/initial_count*100:.1f}%)")
        logger.info(f"      • Registros finales: {len(master):,}")
    
    logger.info(f"   ✅ Master Table: {len(master):,} registros, {len(master.columns)} columnas")
    
    logger.info(f"   ✅ Master Table: {len(master):,} registros, {len(master.columns)} columnas")
    
    return master


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
        datasets['geolocation'].to_sql(
            'geolocation',
            engine_gold,
            schema='dm',
            if_exists='replace',
            index=False,
            method='multi',
            chunksize=5000
        )
        logger.info(f"   ✅ Tabla 'gold.dm.geolocation' creada ({len(datasets['geolocation']):,} registros)")
    
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


@flow(name="Silver → Gold (Master Table)", log_prints=True)
def silver_to_gold():
    """
    Flujo que construye la master table en Gold
    desde todas las tablas de Silver
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("🥇 INICIANDO CONSTRUCCIÓN DE MASTER TABLE - GOLD")
    logger.info("=" * 80)
    
    # 1. Cargar datasets desde Silver
    datasets = load_datasets_from_silver()
    
    # 2. Construir master table
    master_df = build_master_table(datasets)
    
    # 3. Guardar en Gold
    result = save_master_table(master_df, datasets)
    
    logger.info("=" * 80)
    logger.info("✅ MASTER TABLE GOLD COMPLETADA")
    logger.info("=" * 80)
    logger.info(f"📊 Resumen:")
    logger.info(f"   - Registros: {result['total_rows']:,}")
    logger.info(f"   - Columnas: {result['total_columns']}")
    if result.get('target_stats'):
        logger.info(f"   - Target 'Delayed_time':")
        logger.info(f"      • Min: {result['target_stats']['min']:.0f} días")
        logger.info(f"      • Max: {result['target_stats']['max']:.0f} días")
        logger.info(f"      • Media: {result['target_stats']['mean']:.1f} días")
    logger.info("=" * 80)
    
    return result


if __name__ == "__main__":
    silver_to_gold()