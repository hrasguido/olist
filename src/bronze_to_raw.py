# src/bronze_to_raw.py
from prefect import flow, task, get_run_logger
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv("/workspace/.env")

SRC = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/olist"
BRONZE = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/bronze"

TABLES = ["customers", "orders", "order_items", "products", "sellers", "order_payments", "order_reviews"]

@task(log_prints=True)
def bronze_table(table: str):
    logger = get_run_logger()
    logger.info(f"Procesando tabla: {table}")
    
    src = create_engine(SRC)
    dst = create_engine(BRONZE)
    
    df = pd.read_sql(f"SELECT * FROM {table}", src)
    df["ingestion_date"] = datetime.utcnow()
    df["source_table"] = table
    
    df.to_sql(table, dst, schema="raw", if_exists="append", index=False)
    logger.info(f"{table} → {len(df):,} filas copiadas a bronze.raw")

@flow(name="OLIST → Bronze (Raw + Auditoría)")
def olist_to_bronze():
    for t in TABLES:
        bronze_table(t)
    print("BRONZE COMPLETADO CON ÉXITO")

if __name__ == "__main__":
    olist_to_bronze()