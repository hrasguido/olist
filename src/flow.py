# src/flow.py
"""
Flujo Maestro ETL - Arquitectura Medallion
Orquesta la ejecución secuencial: Bronze → Silver → Gold
"""
from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any

# Importar flujos de cada capa
from etl_multi import etl_multi_csv
from bronze_to_raw import olist_to_bronze
from silver_curated import bronze_to_silver
from gold_fact_sales import silver_to_gold
from gold_ml_features import build_ml_pipeline

load_dotenv("/workspace/.env")

# Configuración de conexiones
CONNECTIONS = {
    "olist": f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/olist",
    "bronze": f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/bronze",
    "silver": f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/silver",
    "gold": f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/gold"
}


@task(name="Validar Conexión BD", retries=3, retry_delay_seconds=5)
def validate_connection(db_name: str, conn_string: str) -> bool:
    """Valida que la base de datos esté accesible"""
    logger = get_run_logger()
    
    try:
        engine = create_engine(conn_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            logger.info(f"✅ Conexión exitosa a BD '{db_name}'")
            return True
    except SQLAlchemyError as e:
        logger.error(f"❌ Error conectando a '{db_name}': {str(e)}")
        raise
    except Exception as e:
        logger.error(f"❌ Error inesperado en '{db_name}': {str(e)}")
        raise


@task(name="Ejecutar Capa con Manejo de Errores")
def execute_layer_safe(layer_name: str, flow_func, **kwargs) -> Dict[str, Any]:
    """
    Ejecuta un flujo de capa con manejo robusto de errores
    
    Args:
        layer_name: Nombre de la capa (Bronze/Silver/Gold)
        flow_func: Función del flujo a ejecutar
        **kwargs: Argumentos adicionales para el flujo
    
    Returns:
        Dict con status, timestamp y mensaje
    """
    logger = get_run_logger()
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"🚀 Iniciando capa: {layer_name}")
        
        # Ejecutar el flujo
        result = flow_func(**kwargs)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Capa {layer_name} completada en {duration:.2f}s")
        
        return {
            "layer": layer_name,
            "status": "success",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "message": f"Capa {layer_name} ejecutada exitosamente"
        }
        
    except SQLAlchemyError as e:
        logger.error(f"❌ Error de base de datos en {layer_name}: {str(e)}")
        return {
            "layer": layer_name,
            "status": "failed",
            "error_type": "SQLAlchemyError",
            "error_message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except FileNotFoundError as e:
        logger.error(f"❌ Archivo no encontrado en {layer_name}: {str(e)}")
        return {
            "layer": layer_name,
            "status": "failed",
            "error_type": "FileNotFoundError",
            "error_message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error inesperado en {layer_name}: {type(e).__name__} - {str(e)}")
        return {
            "layer": layer_name,
            "status": "failed",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@task(name="Verificar Resultados de Capa")
def verify_layer_results(layer_name: str, result: Dict[str, Any]) -> bool:
    """Verifica si una capa se ejecutó correctamente"""
    logger = get_run_logger()
    
    if result["status"] == "success":
        logger.info(f"✅ Verificación exitosa: {layer_name}")
        return True
    else:
        logger.error(f"❌ Verificación fallida: {layer_name}")
        logger.error(f"   Tipo de error: {result.get('error_type', 'Unknown')}")
        logger.error(f"   Mensaje: {result.get('error_message', 'No message')}")
        return False


@flow(
    name="🏗️ ETL Maestro - Arquitectura Medallion",
    description="Orquesta la ejecución completa: Ingesta → Bronze → Silver → Gold",
    task_runner=SequentialTaskRunner(),
    log_prints=True
)
def master_etl_pipeline(
    skip_ingestion: bool = False,
    skip_bronze: bool = False,
    skip_silver: bool = False,
    skip_gold: bool = False,
    skip_ml: bool = False  # NUEVO
):
    """
    Flujo maestro que ejecuta el pipeline completo ETL
    
    Args:
        skip_ingestion: Si True, omite la carga inicial de CSVs
        skip_bronze: Si True, omite la capa Bronze
        skip_silver: Si True, omite la capa Silver
        skip_gold: Si True, omite la capa Gold
    """
    logger = get_run_logger()
    pipeline_start = datetime.utcnow()
    results = []
    
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO PIPELINE ETL MAESTRO - ARQUITECTURA MEDALLION")
    logger.info("=" * 80)
    
    try:
        # ============================================================
        # FASE 0: VALIDACIÓN DE CONEXIONES
        # ============================================================
        logger.info("\n📡 FASE 0: Validando conexiones a bases de datos...")
        
        for db_name, conn_string in CONNECTIONS.items():
            try:
                validate_connection(db_name, conn_string)
            except Exception as e:
                logger.error(f"❌ No se pudo conectar a '{db_name}'. Abortando pipeline.")
                raise
        
        logger.info("✅ Todas las conexiones validadas correctamente\n")
        
        # ============================================================
        # FASE 1: INGESTA DE DATOS (CSV → BD Olist)
        # ============================================================
        if not skip_ingestion:
            logger.info("📥 FASE 1: Ingesta de datos CSV → BD Olist")
            ingestion_result = execute_layer_safe(
                "Ingestion (CSV → Olist)",
                etl_multi_csv
            )
            results.append(ingestion_result)
            
            if not verify_layer_results("Ingestion", ingestion_result):
                logger.error("❌ Ingesta falló. Abortando pipeline.")
                raise Exception("Fallo en fase de ingesta")
        else:
            logger.info("⏭️  FASE 1: Ingesta omitida (skip_ingestion=True)\n")
        
        # ============================================================
        # FASE 2: CAPA BRONZE (Olist → Bronze.Raw)
        # ============================================================
        if not skip_bronze:
            logger.info("🥉 FASE 2: Capa Bronze (Olist → Bronze.Raw)")
            bronze_result = execute_layer_safe(
                "Bronze",
                olist_to_bronze
            )
            results.append(bronze_result)
            
            if not verify_layer_results("Bronze", bronze_result):
                logger.error("❌ Capa Bronze falló. Abortando pipeline.")
                raise Exception("Fallo en capa Bronze")
        else:
            logger.info("⏭️  FASE 2: Capa Bronze omitida (skip_bronze=True)\n")
        
        # ============================================================
        # FASE 3: CAPA SILVER (Bronze → Silver.Curated)
        # ============================================================
        if not skip_silver:
            logger.info("🥈 FASE 3: Capa Silver (Bronze → Silver.Curated)")
            silver_result = execute_layer_safe(
                "Silver",
                bronze_to_silver
            )
            results.append(silver_result)
            
            if not verify_layer_results("Silver", silver_result):
                logger.error("❌ Capa Silver falló. Abortando pipeline.")
                raise Exception("Fallo en capa Silver")
        else:
            logger.info("⏭️  FASE 3: Capa Silver omitida (skip_silver=True)\n")
        
        # ============================================================
        # FASE 4: CAPA GOLD (Silver → Gold.DM)
        # ============================================================
        if not skip_gold:
            logger.info("🥇 FASE 4: Capa Gold (Silver → Gold.DM)")
            gold_result = execute_layer_safe(
                "Gold",
                silver_to_gold
            )
            results.append(gold_result)
            
            if not verify_layer_results("Gold", gold_result):
                logger.error("❌ Capa Gold falló.")
                raise Exception("Fallo en capa Gold")
        else:
            logger.info("⏭️  FASE 4: Capa Gold omitida (skip_gold=True)\n")
        
        # ============================================================
        # FASE 5: CAPA GOLD ML (Silver → Gold.ML)
        # ============================================================
        if not skip_ml:
            logger.info("🤖 FASE 5: Capa Gold ML (Silver → Gold.ML Master Table)")
            ml_result = execute_layer_safe(
                "Gold ML",
                build_ml_pipeline
            )
            results.append(ml_result)
            
            if not verify_layer_results("Gold ML", ml_result):
                logger.warning("⚠️  Capa Gold ML falló (no crítico).")
        else:
            logger.info("⏭️  FASE 5: Capa Gold ML omitida (skip_ml=True)\n")

        # ============================================================
        # RESUMEN FINAL
        # ============================================================
        pipeline_end = datetime.utcnow()
        total_duration = (pipeline_end - pipeline_start).total_seconds()
        
        logger.info("=" * 80)
        logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Duración total: {total_duration:.2f} segundos")
        logger.info(f"📊 Capas ejecutadas: {len(results)}")
        
        for result in results:
            status_icon = "✅" if result["status"] == "success" else "❌"
            logger.info(f"   {status_icon} {result['layer']}: {result['status']}")
        
        logger.info("=" * 80)
        
        return {
            "status": "success",
            "total_duration_seconds": total_duration,
            "layers_executed": len(results),
            "results": results
        }
        
    except Exception as e:
        pipeline_end = datetime.utcnow()
        total_duration = (pipeline_end - pipeline_start).total_seconds()
        
        logger.error("=" * 80)
        logger.error("❌ PIPELINE FALLÓ")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duración hasta fallo: {total_duration:.2f} segundos")
        logger.error(f"🔥 Error: {type(e).__name__} - {str(e)}")
        logger.error("=" * 80)
        
        return {
            "status": "failed",
            "total_duration_seconds": total_duration,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "results": results
        }


@flow(name="ETL ML - Solo Master Table")
def quick_ml_refresh():
    """Flujo rápido que solo actualiza la master table ML"""
    logger = get_run_logger()
    logger.info("🤖 Ejecutando actualización rápida de ML Master Table...")
    
    return master_etl_pipeline(
        skip_ingestion=True,
        skip_bronze=True,
        skip_silver=True,
        skip_gold=True,
        skip_ml=False
    )


if __name__ == "__main__":
    # Ejecutar pipeline completo
    master_etl_pipeline()
