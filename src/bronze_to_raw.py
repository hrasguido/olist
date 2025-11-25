# src/bronze_to_raw.py
"""
Capa Bronze - Raw Data con Limpieza
Ingesta incremental desde BD Olist a Bronze con limpieza y auditoría
"""
from prefect import flow, task, get_run_logger
from sqlalchemy import create_engine, text, inspect
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple

load_dotenv("/workspace/.env")

SRC = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/olist"
BRONZE = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/bronze"

TABLES = ["customers", "orders", "order_items", "products", "sellers", "order_payments", "order_reviews", "geolocation"]

# Definir columnas PK por tabla para identificar registros únicos
PRIMARY_KEYS = {
    "customers": ["customer_id"],
    "orders": ["order_id"],
    "order_items": ["order_id", "order_item_id"],
    "products": ["product_id"],
    "sellers": ["seller_id"],
    "order_payments": ["order_id", "payment_sequential"],
    "order_reviews": ["review_id"],
    "geolocation": ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]
}


# ============================================================
# FUNCIONES DE LIMPIEZA ESPECÍFICAS POR TABLA
# ============================================================

def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza específica para tabla orders"""
    logger = get_run_logger()
    df = df.copy()
    
    initial_rows = len(df)
    
    # Filtrar solo órdenes entregadas
    df = df[df['order_status'] == 'delivered']
    delivered_filtered = initial_rows - len(df)
    if delivered_filtered > 0:
        logger.info(f"      🔍 Filtradas {delivered_filtered:,} órdenes no entregadas")
    
    # Eliminar filas con nulos
    rows_before = len(df)
    df = df.dropna()
    nulls_removed = rows_before - len(df)
    if nulls_removed > 0:
        logger.info(f"      🗑️  Eliminadas {nulls_removed:,} filas con nulos")
    
    # Convertir fechas
    date_cols = [
        'order_purchase_timestamp', 'order_approved_at',
        'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df


def clean_order_items(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza específica para tabla order_items"""
    logger = get_run_logger()
    df = df.copy()
    
    # Convertir fecha de límite de envío
    if 'shipping_limit_date' in df.columns:
        df['shipping_limit_date'] = pd.to_datetime(df['shipping_limit_date'], errors='coerce')
        logger.info(f"      📅 Convertida columna 'shipping_limit_date' a datetime")
    
    return df


def clean_order_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza específica para tabla order_payments"""
    logger = get_run_logger()
    df = df.copy()
    
    initial_rows = len(df)
    
    # Eliminar pagos con 0 cuotas
    df = df[df['payment_installments'] != 0]
    installments_removed = initial_rows - len(df)
    if installments_removed > 0:
        logger.info(f"      🗑️  Eliminados {installments_removed:,} pagos con 0 cuotas")
    
    # Eliminar pagos "not_defined" con valor 0
    notdefined_0 = df[(df['payment_value'] == 0) & (df['payment_type'] == 'not_defined')]
    if len(notdefined_0) > 0:
        logger.info(f"      🗑️  Eliminados {len(notdefined_0):,} pagos 'not_defined' con valor 0")
        df.drop(notdefined_0.index, inplace=True)
    
    # Formatear tipo de pago
    if 'payment_type' in df.columns:
        df['payment_type'] = df['payment_type'].str.replace('_', ' ').str.title()
        logger.info(f"      ✨ Formateados tipos de pago (ej: 'credit_card' → 'Credit Card')")
    
    return df


def clean_order_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza específica para tabla order_reviews"""
    logger = get_run_logger()
    df = df.copy()
    
    # Eliminar columnas de comentarios
    cols_to_drop = ['review_comment_title', 'review_comment_message']
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    
    if existing_cols:
        df = df.drop(columns=existing_cols)
        logger.info(f"      🗑️  Eliminadas columnas: {', '.join(existing_cols)}")
    
    return df


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza específica para tabla products"""
    logger = get_run_logger()
    df = df.copy()
    
    initial_rows = len(df)
    
    # Eliminar productos con nulos
    rows_before = len(df)
    df.dropna(inplace=True)
    nulls_removed = rows_before - len(df)
    if nulls_removed > 0:
        logger.info(f"      🗑️  Eliminados {nulls_removed:,} productos con nulos")
    
    # Eliminar productos con peso 0
    if 'product_weight_g' in df.columns:
        zero_weight = df[df['product_weight_g'] == 0]
        if len(zero_weight) > 0:
            logger.info(f"      🗑️  Eliminados {len(zero_weight):,} productos con peso 0")
            df.drop(zero_weight.index, inplace=True)
    
    return df


# Mapeo de funciones de limpieza por tabla
CLEANING_FUNCTIONS = {
    "orders": clean_orders,
    "order_items": clean_order_items,
    "order_payments": clean_order_payments,
    "order_reviews": clean_order_reviews,
    "products": clean_products
}


def get_existing_keys(engine, table: str, pk_columns: List[str]) -> set:
    """
    Obtiene las claves primarias que ya existen en Bronze
    
    Args:
        engine: SQLAlchemy engine
        table: Nombre de la tabla
        pk_columns: Lista de columnas que forman la PK
    
    Returns:
        Set de tuplas con las PKs existentes
    """
    logger = get_run_logger()
    
    # Verificar si la tabla existe en Bronze
    inspector = inspect(engine)
    if not inspector.has_table(table, schema="raw"):
        logger.info(f"   ℹ️  Tabla '{table}' no existe en Bronze. Primera carga.")
        return set()
    
    # Construir query para obtener PKs existentes
    pk_select = ", ".join(pk_columns)
    query = f"SELECT DISTINCT {pk_select} FROM raw.{table}"
    
    try:
        df_existing = pd.read_sql(query, engine)
        
        # Convertir a set de tuplas
        if len(pk_columns) == 1:
            existing_keys = set(df_existing[pk_columns[0]].values)
        else:
            existing_keys = set(df_existing[pk_columns].itertuples(index=False, name=None))
        
        logger.info(f"   📊 Registros existentes en Bronze: {len(existing_keys):,}")
        return existing_keys
        
    except Exception as e:
        logger.warning(f"   ⚠️  Error leyendo PKs existentes: {str(e)}. Asumiendo tabla vacía.")
        return set()


def filter_new_records(df: pd.DataFrame, existing_keys: set, pk_columns: List[str]) -> pd.DataFrame:
    """
    Filtra el DataFrame para mantener solo registros nuevos
    
    Args:
        df: DataFrame con todos los datos
        existing_keys: Set de PKs existentes
        pk_columns: Lista de columnas PK
    
    Returns:
        DataFrame solo con registros nuevos
    """
    if not existing_keys:
        return df
    
    # Crear columna temporal con la PK compuesta
    if len(pk_columns) == 1:
        df['_temp_pk'] = df[pk_columns[0]]
        mask = ~df['_temp_pk'].isin(existing_keys)
    else:
        df['_temp_pk'] = df[pk_columns].apply(tuple, axis=1)
        mask = ~df['_temp_pk'].isin(existing_keys)
    
    # Filtrar y eliminar columna temporal
    df_new = df[mask].drop(columns=['_temp_pk'])
    
    return df_new


@task(log_prints=True, retries=2, retry_delay_seconds=5)
def bronze_table_incremental(table: str) -> Dict:
    """
    Procesa una tabla de forma incremental desde Olist a Bronze con limpieza
    
    Args:
        table: Nombre de la tabla a procesar
    
    Returns:
        Dict con estadísticas de procesamiento
    """
    logger = get_run_logger()
    logger.info(f"📋 Procesando tabla: {table}")
    
    src_engine = create_engine(SRC)
    dst_engine = create_engine(BRONZE)
    
    # Obtener columnas PK
    pk_columns = PRIMARY_KEYS.get(table, [])
    if not pk_columns:
        logger.warning(f"   ⚠️  No se definió PK para '{table}'. Se cargarán todos los registros.")
    
    # Leer datos desde origen
    df_source = pd.read_sql(f"SELECT * FROM {table}", src_engine)
    total_source = len(df_source)
    logger.info(f"   📥 Registros en origen (Olist): {total_source:,}")
    
    if total_source == 0:
        logger.warning(f"   ⚠️  Tabla '{table}' está vacía en origen. Omitiendo.")
        return {
            "table": table,
            "source_records": 0,
            "existing_records": 0,
            "new_records": 0,
            "loaded_records": 0,
            "cleaned_records": 0,
            "status": "empty_source"
        }
    
    # ============================================================
    # APLICAR LIMPIEZA ESPECÍFICA POR TABLA
    # ============================================================
    rows_before_cleaning = len(df_source)
    
    if table in CLEANING_FUNCTIONS:
        logger.info(f"   🧹 Aplicando limpieza específica para '{table}'...")
        df_source = CLEANING_FUNCTIONS[table](df_source)
        rows_after_cleaning = len(df_source)
        cleaned_count = rows_before_cleaning - rows_after_cleaning
        if cleaned_count > 0:
            logger.info(f"      ✅ Limpieza completada: {cleaned_count:,} filas removidas")
    else:
        cleaned_count = 0
    
    # Obtener registros existentes en Bronze
    existing_keys = get_existing_keys(dst_engine, table, pk_columns) if pk_columns else set()
    existing_count = len(existing_keys)
    
    # Filtrar solo registros nuevos
    if pk_columns:
        df_new = filter_new_records(df_source, existing_keys, pk_columns)
        new_count = len(df_new)
        
        if new_count == 0:
            logger.info(f"   ✅ No hay registros nuevos para '{table}'. Tabla actualizada.")
            return {
                "table": table,
                "source_records": total_source,
                "existing_records": existing_count,
                "new_records": 0,
                "loaded_records": 0,
                "cleaned_records": cleaned_count,
                "status": "up_to_date"
            }
        
        logger.info(f"   🆕 Registros nuevos detectados: {new_count:,}")
    else:
        # Si no hay PK definida, cargar todo
        df_new = df_source
        new_count = len(df_source)
        logger.info(f"   ⚠️  Cargando todos los registros (sin validación de duplicados)")
    
    # Añadir columnas de auditoría
    df_new["ingestion_date"] = datetime.utcnow()
    df_new["source_table"] = table
    
    # Cargar a Bronze
    df_new.to_sql(
        table, 
        dst_engine, 
        schema="raw", 
        if_exists="append", 
        index=False,
        method='multi',
        chunksize=5000
    )
    
    logger.info(f"   ✅ {new_count:,} registros nuevos cargados a bronze.raw.{table}")
    
    return {
        "table": table,
        "source_records": total_source,
        "existing_records": existing_count,
        "new_records": new_count,
        "loaded_records": new_count,
        "cleaned_records": cleaned_count,
        "status": "success"
    }


@flow(name="OLIST → Bronze (Incremental + Limpieza + Auditoría)", log_prints=True)
def olist_to_bronze():
    """
    Flujo principal que carga datos de forma incremental desde Olist a Bronze
    Incluye limpieza de datos y solo carga registros que no existan previamente
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("🥉 INICIANDO CARGA INCREMENTAL A BRONZE CON LIMPIEZA")
    logger.info("=" * 80)
    
    results = []
    total_source = 0
    total_existing = 0
    total_new = 0
    total_loaded = 0
    total_cleaned = 0
    
    for table in TABLES:
        try:
            stats = bronze_table_incremental(table)
            results.append(stats)
            
            total_source += stats["source_records"]
            total_existing += stats["existing_records"]
            total_new += stats["new_records"]
            total_loaded += stats["loaded_records"]
            total_cleaned += stats["cleaned_records"]
            
        except Exception as e:
            logger.error(f"❌ Error procesando tabla '{table}': {str(e)}")
            raise
    
    # Resumen final
    logger.info("=" * 80)
    logger.info("✅ CARGA BRONZE COMPLETADA")
    logger.info("=" * 80)
    logger.info(f"📊 Resumen General:")
    logger.info(f"   - Tablas procesadas: {len(results)}")
    logger.info(f"   - Registros en origen: {total_source:,}")
    logger.info(f"   - Registros limpiados: {total_cleaned:,}")
    logger.info(f"   - Registros ya existentes: {total_existing:,}")
    logger.info(f"   - Registros nuevos: {total_new:,}")
    logger.info(f"   - Registros cargados: {total_loaded:,}")
    logger.info("")
    logger.info("📋 Detalle por tabla:")
    
    for stat in results:
        status_icon = "✅" if stat["status"] == "success" else "ℹ️"
        if stat["status"] == "success":
            logger.info(f"   {status_icon} {stat['table']}: "
                       f"{stat['source_records']:,} origen → "
                       f"{stat['cleaned_records']:,} limpiados → "
                       f"{stat['new_records']:,} nuevos cargados")
        elif stat["status"] == "up_to_date":
            logger.info(f"   {status_icon} {stat['table']}: Actualizada (0 nuevos)")
        elif stat["status"] == "empty_source":
            logger.info(f"   ⚠️  {stat['table']}: Vacía en origen")
    
    logger.info("=" * 80)
    
    return {
        "status": "success",
        "tables_processed": len(results),
        "total_source_records": total_source,
        "total_existing_records": total_existing,
        "total_new_records": total_new,
        "total_loaded_records": total_loaded,
        "total_cleaned_records": total_cleaned,
        "results": results
    }


if __name__ == "__main__":
    olist_to_bronze()