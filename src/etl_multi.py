# src/etl_multi.py
"""
Ingesta CSV → Olist → Bronze → Silver
Carga archivos CSV sin transformaciones y los procesa hasta Silver
"""
from prefect import flow, task, get_run_logger
from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path

# Importar flujos de Bronze y Silver
from bronze_to_raw import olist_to_bronze
from silver_curated import bronze_to_silver
from gold_fact_sales import silver_to_gold

load_dotenv("/workspace/.env")

# Configuración
DATA_DIR = Path("/workspace/data")
TABLE_MAPPING = {
    "olist_orders_last_3_months.csv": "orders",
    "olist_customers_dataset.csv": "customers",
    "olist_order_items_dataset.csv": "order_items",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_geolocation_dataset.csv": "geolocation",
    "product_category_name_translation.csv": "product_category_translation"
}


@task(retries=2, retry_delay_seconds=10)
def load_csv_to_olist(csv_file: Path, table_name: str) -> int:
    """
    Carga un archivo CSV directamente a la BD Olist sin transformaciones
    
    Args:
        csv_file: Ruta al archivo CSV
        table_name: Nombre de la tabla destino
    
    Returns:
        Número de filas cargadas
    """
    logger = get_run_logger()
    logger.info(f"📋 Procesando: {csv_file.name} → {table_name}")
    
    # Leer CSV sin transformaciones
    df = pd.read_csv(csv_file)
    rows = len(df)
    logger.info(f"   📥 Leídos {rows:,} registros")
    
    # Crear conexión
    engine = create_engine(
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/{os.getenv('POSTGRES_DB')}"
    )
    
    # Cargar directamente a BD
    df.to_sql(
        table_name,
        engine,
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=5000
    )
    
    # Verificar carga
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    
    logger.info(f"   ✅ {count:,} filas cargadas en olist.{table_name}")
    return count


@flow(name="ETL Multi-CSV → Olist → Bronze → Silver", log_prints=True)
def etl_multi_csv():
    """
    Flujo completo que:
    1. Carga archivos CSV a BD Olist (sin transformaciones)
    2. Mueve datos de Olist a Bronze (incremental)
    3. Procesa datos de Bronze a Silver (limpieza y validación)
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO ETL: CSV → OLIST → BRONZE → SILVER")
    logger.info("=" * 80)
    
    # ============================================================
    # FASE 1: CARGAR CSVs A OLIST (SIN TRANSFORMACIONES)
    # ============================================================
    logger.info("\n📥 FASE 1: Cargando archivos CSV a BD Olist (raw)...")
    
    results = []
    total_loaded = 0
    files_processed = 0
    files_skipped = 0
    
    for csv_file, table_name in TABLE_MAPPING.items():
        file_path = DATA_DIR / csv_file
        
        if not file_path.exists():
            logger.warning(f"⚠️  Archivo no encontrado: {csv_file}")
            files_skipped += 1
            continue
        
        try:
            count = load_csv_to_olist(file_path, table_name)
            
            results.append({
                "file": csv_file,
                "table": table_name,
                "rows": count,
                "status": "success"
            })
            
            total_loaded += count
            files_processed += 1
            
        except Exception as e:
            logger.error(f"❌ Error procesando {csv_file}: {str(e)}")
            raise
    
    # Resumen Fase 1
    logger.info("\n" + "-" * 80)
    logger.info("✅ FASE 1 COMPLETADA")
    logger.info(f"   - Archivos procesados: {files_processed}")
    logger.info(f"   - Archivos omitidos: {files_skipped}")
    logger.info(f"   - Total registros en Olist: {total_loaded:,}")
    logger.info("-" * 80)
    
    # ============================================================
    # FASE 2: MOVER DE OLIST A BRONZE (INCREMENTAL)
    # ============================================================
    logger.info("\n🥉 FASE 2: Moviendo datos de Olist a Bronze (incremental)...")
    
    try:
        bronze_result = olist_to_bronze()
        
        if bronze_result["status"] != "success":
            logger.error("❌ Carga a Bronze falló. Abortando proceso.")
            return {
                "status": "failed",
                "phase": "bronze",
                "files_processed": files_processed,
                "total_loaded": total_loaded,
                "bronze_result": bronze_result
            }
        
        logger.info(f"\n✅ Fase 2 completada: {bronze_result['total_new_records']:,} registros nuevos en Bronze")
        
    except Exception as e:
        logger.error(f"❌ Error en carga a Bronze: {str(e)}")
        raise
    
    # ============================================================
    # FASE 3: PROCESAR DE BRONZE A SILVER (LIMPIEZA)
    # ============================================================
    logger.info("\n🥈 FASE 3: Procesando datos de Bronze a Silver (limpieza y validación)...")
    
    try:
        silver_result = bronze_to_silver()
        
        if silver_result["status"] != "success":
            logger.error("❌ Procesamiento a Silver falló.")
            return {
                "status": "failed",
                "phase": "silver",
                "files_processed": files_processed,
                "total_loaded": total_loaded,
                "bronze_result": bronze_result,
                "silver_result": silver_result
            }
        
        logger.info(f"\n✅ Fase 3 completada: {silver_result['total_final_rows']:,} registros limpios en Silver")
        
    except Exception as e:
        logger.error(f"❌ Error en procesamiento a Silver: {str(e)}")
        raise

    # ============================================================
    # FASE 4: PROCESAR DE SILVER A GOLD (MODELADO)
    # ============================================================
    logger.info("\n🥇 FASE 4: Procesando datos de Silver a Gold (modelado y validación)...")
    
    try:
        gold_result = silver_to_gold()
        
        if gold_result["status"] != "success":
            logger.error("❌ Procesamiento a Gold falló.")
            return {
                "status": "failed",
                "phase": "gold",
                "files_processed": files_processed,
                "total_loaded": total_loaded,
                "bronze_result": bronze_result,
                "silver_result": silver_result,
                "gold_result": gold_result
            }
        
        logger.info(f"\n✅ Fase 4 completada: {gold_result['total_rows']:,} registros en Master Table Gold")
        
    except Exception as e:
        logger.error(f"❌ Error en procesamiento a Gold: {str(e)}")
        raise
    
    # ============================================================
    # RESUMEN FINAL
    # ============================================================
    logger.info("\n" + "=" * 80)
    logger.info("✅ ETL COMPLETO FINALIZADO EXITOSAMENTE")
    logger.info("=" * 80)
    logger.info("📊 Resumen General:")
    logger.info("")
    logger.info("   FASE 1 - Ingesta CSV → Olist:")
    logger.info(f"      • Archivos procesados: {files_processed}")
    logger.info(f"      • Registros cargados: {total_loaded:,}")
    logger.info("")
    logger.info("   FASE 2 - Olist → Bronze:")
    logger.info(f"      • Tablas procesadas: {bronze_result['tables_processed']}")
    logger.info(f"      • Registros en origen: {bronze_result['total_source_records']:,}")
    logger.info(f"      • Registros nuevos cargados: {bronze_result['total_new_records']:,}")
    logger.info(f"      • Registros ya existentes: {bronze_result['total_existing_records']:,}")
    logger.info("")
    logger.info("   FASE 3 - Bronze → Silver:")
    logger.info(f"      • Tablas procesadas: {silver_result['tables_processed']}")
    logger.info(f"      • Registros iniciales: {silver_result['total_initial_rows']:,}")
    logger.info(f"      • Registros finales: {silver_result['total_final_rows']:,}")
    logger.info(f"      • Registros limpiados: {silver_result['total_cleaned_rows']:,}")
    logger.info("")
    logger.info("   FASE 4 - Silver → Gold:")
    logger.info(f"      • Registros en Master Table: {gold_result['total_rows']:,}")
    logger.info(f"      • Columnas: {gold_result['total_columns']}")
    logger.info("")
    logger.info("=" * 80)
    
    return {
        "status": "success",
        "phase_1": {
            "files_processed": files_processed,
            "files_skipped": files_skipped,
            "total_loaded": total_loaded,
            "results": results
        },
        "phase_2": bronze_result,
        "phase_3": silver_result,
        "phase_4": gold_result  # NUEVO
    }


if __name__ == "__main__":
    etl_multi_csv()