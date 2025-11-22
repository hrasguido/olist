# src/etl_multi.py
from prefect import flow, task, get_run_logger
from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv("/workspace/.env")

# Configuración
DATA_DIR = Path("/workspace/data")
TABLE_MAPPING = {
    "olist_orders_dataset.csv": "orders",
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
def extract(csv_file: Path) -> pd.DataFrame:
    logger = get_run_logger()
    df = pd.read_csv(csv_file)
    logger.info(f"Extraídos {len(df)} registros de {csv_file.name}")
    return df

@task
def transform(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    logger = get_run_logger()
    
    # Limpieza básica
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Fechas
    date_cols = [col for col in df.columns if "timestamp" in col or "date" in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    logger.info(f"Transformados {len(df)} registros para tabla `{table_name}`")
    return df

@task
def load(df: pd.DataFrame, table_name: str):
    logger = get_run_logger()
    
    engine = create_engine(
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/{os.getenv('POSTGRES_DB')}"
    )
    
    df.to_sql(
        table_name,
        engine,
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=5000
    )
    
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    
    logger.info(f"Tabla `{table_name}` → {count} filas cargadas")

@flow(name="ETL Multi-CSV → PostgreSQL")
def etl_multi_csv():
    for csv_file, table_name in TABLE_MAPPING.items():
        file_path = DATA_DIR / csv_file
        if not file_path.exists():
            get_run_logger().warning(f"Archivo no encontrado: {csv_file}")
            continue
            
        raw = extract(file_path)
        clean = transform(raw, table_name)
        load(clean, table_name)

if __name__ == "__main__":
    etl_multi_csv()