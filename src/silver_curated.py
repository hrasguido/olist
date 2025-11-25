# src/silver_curated.py
"""
Capa Silver - Master Table Unificada
Crea una tabla maestra consolidando todas las fuentes desde Bronze
"""
from prefect import flow, task, get_run_logger
from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Dict

load_dotenv("/workspace/.env")

BRONZE_CONN = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/bronze"
SILVER_CONN = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/silver"

engine_bronze = create_engine(BRONZE_CONN)
engine_silver = create_engine(SILVER_CONN)


@task(log_prints=True, retries=2, retry_delay_seconds=5)
def load_datasets_from_bronze() -> Dict[str, pd.DataFrame]:
    """Carga todos los datasets desde Bronze."""
    logger = get_run_logger()
    logger.info("📥 Cargando datasets desde Bronze...")
    
    datasets = {}
    tables = [
        'customers', 'geolocation', 'order_items', 'order_payments',
        'order_reviews', 'orders', 'products', 'sellers', 'product_category_translation'
    ]
    
    for table in tables:
        try:
            logger.info(f"   Cargando: {table}")
            df = pd.read_sql(f"SELECT * FROM raw.{table}", engine_bronze)
            datasets[table] = df
            logger.info(f"   ✅ {table}: {len(df):,} registros")
        except Exception as e:
            logger.warning(f"   ⚠️  Error cargando {table}: {str(e)}")
            datasets[table] = pd.DataFrame()
    
    return datasets


@task(log_prints=True)
def build_master_table(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Construye la master table unificada desde todas las fuentes.
    """
    logger = get_run_logger()
    logger.info("🔨 Construyendo Master Table...")
    
    # Verificar que tenemos los datasets principales
    required = ['orders', 'order_items', 'customers', 'products', 'sellers']
    for table in required:
        if table not in datasets or datasets[table].empty:
            raise ValueError(f"Dataset requerido '{table}' no disponible")
    
    # Base: orders
    master = datasets['orders'].copy()
    initial_rows = len(master)
    logger.info(f"   Base (orders): {initial_rows:,} registros")
    
    # Join con customers
    if not datasets['customers'].empty:
        master = master.merge(
            datasets['customers'],
            on='customer_id',
            how='left',
            suffixes=('', '_customer')
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
            suffixes=('', '_product')
        )
        logger.info(f"   + products: {len(master):,} registros")
    
    # Join con sellers
    if not datasets['sellers'].empty:
        master = master.merge(
            datasets['sellers'],
            on='seller_id',
            how='left',
            suffixes=('', '_seller')
        )
        logger.info(f"   + sellers: {len(master):,} registros")
    
    # Join con order_payments (agregado por order_id)
    if not datasets['order_payments'].empty:
        payments_agg = datasets['order_payments'].groupby('order_id').agg({
            'payment_sequential': 'count',
            'payment_type': lambda x: x.mode()[0] if not x.mode().empty else 'unknown',
            'payment_installments': 'max',
            'payment_value': 'sum'
        }).reset_index()
        
        payments_agg.columns = ['order_id', 'payment_count', 'payment_type', 
                                'payment_installments', 'payment_value']
        
        master = master.merge(
            payments_agg,
            on='order_id',
            how='left',
            suffixes=('', '_payment')
        )
        logger.info(f"   + order_payments (agg): {len(master):,} registros")
    
    # Join con order_reviews (agregado por order_id)
    if not datasets['order_reviews'].empty:
        reviews_agg = datasets['order_reviews'].groupby('order_id').agg({
            'review_score': 'mean',
            'review_comment_title': 'count'
        }).reset_index()
        
        reviews_agg.columns = ['order_id', 'review_score', 'review_count']
        
        master = master.merge(
            reviews_agg,
            on='order_id',
            how='left',
            suffixes=('', '_review')
        )
        logger.info(f"   + order_reviews (agg): {len(master):,} registros")
    
    # Join con category_translation (si existe)
    if 'product_category_translation' in datasets and not datasets['product_category_translation'].empty:
        master = master.merge(
            datasets['product_category_translation'],
            on='product_category_name',
            how='left',
            suffixes=('', '_translation')
        )
        logger.info(f"   + category_translation: {len(master):,} registros")
    
    logger.info(f"   ✅ Master Table: {len(master):,} registros, {len(master.columns)} columnas")
    
    return master


@task(log_prints=True)
def save_master_table(master_df: pd.DataFrame) -> Dict:
    """Guarda la master table en Silver."""
    logger = get_run_logger()
    logger.info("💾 Guardando Master Table en Silver...")
    
    initial_rows = len(master_df)
    
    # Filtrar solo pedidos entregados para el análisis
    if 'order_status' in master_df.columns:
        master_clean = master_df[master_df['order_status'] == 'delivered'].copy()
        logger.info(f"   Filtrado: solo pedidos 'delivered'")
        logger.info(f"   Registros filtrados: {len(master_clean):,} de {initial_rows:,}")
    else:
        master_clean = master_df.copy()
    
    final_rows = len(master_clean)
    
    # Guardar en Silver
    master_clean.to_sql(
        'master_table',
        engine_silver,
        schema='curated',
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=5000
    )
    
    logger.info(f"   ✅ Tabla 'silver.curated.master_table' creada")
    logger.info(f"   📊 {final_rows:,} registros, {len(master_clean.columns)} columnas")
    
    return {
        "status": "success",
        "total_rows": final_rows,
        "total_columns": len(master_clean.columns),
        "rows_filtered": initial_rows - final_rows
    }


@flow(name="Bronze → Silver (Master Table)", log_prints=True)
def bronze_to_silver():
    """
    Flujo que crea una master table unificada en Silver
    desde todas las tablas de Bronze
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("🥈 INICIANDO CONSTRUCCIÓN DE MASTER TABLE - SILVER")
    logger.info("=" * 80)
    
    # 1. Cargar datasets desde Bronze
    datasets = load_datasets_from_bronze()
    
    # 2. Construir master table
    master_df = build_master_table(datasets)
    
    # 3. Guardar en Silver
    result = save_master_table(master_df)
    
    logger.info("=" * 80)
    logger.info("✅ MASTER TABLE SILVER COMPLETADA")
    logger.info("=" * 80)
    logger.info(f"📊 Resumen:")
    logger.info(f"   - Registros: {result['total_rows']:,}")
    logger.info(f"   - Columnas: {result['total_columns']}")
    logger.info("=" * 80)
    
    return {
        "status": "success",
        "tables_processed": 1,
        "total_initial_rows": result['total_rows'] + result['rows_filtered'],
        "total_final_rows": result['total_rows'],
        "total_cleaned_rows": result['rows_filtered'],
        "results": [result]
    }


if __name__ == "__main__":
    bronze_to_silver()