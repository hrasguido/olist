# src/silver_curated.py
"""
Capa Silver - Curated Data
Validación y normalización de datos desde Bronze (limpieza ya aplicada)
"""
from prefect import flow, task, get_run_logger
from sqlalchemy import create_engine, types
import pandas as pd
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv("/workspace/.env")

BRONZE_CONN = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/bronze"
SILVER_CONN = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/silver"

engine_bronze = create_engine(BRONZE_CONN)
engine_silver = create_engine(SILVER_CONN)

TABLES = [
    "customers", "sellers", "products", "geolocation",
    "orders", "order_items", "order_payments", "order_reviews"
]


@task(log_prints=True, retries=2, retry_delay_seconds=5)
def silver_table(table_name: str) -> Dict:
    """
    Procesa una tabla desde Bronze a Silver
    Bronze ya tiene datos limpios, Silver solo normaliza y valida
    
    Args:
        table_name: Nombre de la tabla a procesar
    
    Returns:
        Dict con estadísticas de procesamiento
    """
    logger = get_run_logger()
    logger.info(f"📋 Procesando tabla: {table_name}")
    
    # Leer datos desde Bronze (ya limpios)
    df = pd.read_sql(f"SELECT * FROM raw.{table_name}", engine_bronze)
    initial_rows = len(df)
    logger.info(f"   Filas desde Bronze: {initial_rows:,}")
    
    # Cargar directamente a Silver (datos ya están limpios)
    df.to_sql(
        table_name, 
        engine_silver, 
        schema="curated", 
        if_exists="replace", 
        index=False,
        method='multi',
        chunksize=5000
    )
    
    logger.info(f"   ✅ Tabla {table_name} → {initial_rows:,} filas cargadas en silver.curated")
    
    return {
        "table": table_name,
        "rows_processed": initial_rows
    }


@flow(name="Bronze → Silver (Curated)", log_prints=True)
def bronze_to_silver():
    """
    Flujo que mueve datos limpios de Bronze a Silver
    La limpieza ya se aplicó en Bronze
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("🥈 INICIANDO PROCESAMIENTO SILVER - CURATED DATA")
    logger.info("=" * 80)
    
    results = []
    total_rows = 0
    
    for table in TABLES:
        try:
            stats = silver_table(table)
            results.append(stats)
            total_rows += stats["rows_processed"]
        except Exception as e:
            logger.error(f"❌ Error procesando tabla '{table}': {str(e)}")
            raise
    
    logger.info("=" * 80)
    logger.info("✅ PROCESAMIENTO SILVER COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"📊 Resumen General:")
    logger.info(f"   - Tablas procesadas: {len(results)}")
    logger.info(f"   - Total registros: {total_rows:,}")
    logger.info("=" * 80)
    
    return {
        "status": "success",
        "tables_processed": len(results),
        "total_initial_rows": total_rows,
        "total_final_rows": total_rows,
        "total_cleaned_rows": 0,
        "results": results
    }


if __name__ == "__main__":
    bronze_to_silver()