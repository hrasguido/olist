# src/etl_silver.py
from prefect import flow, task
from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv("/workspace/.env")

BRONZE_CONN = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/bronze"
SILVER_CONN = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/silver"

engine_bronze = create_engine(BRONZE_CONN)
engine_silver = create_engine(SILVER_CONN)

TABLES = [
    "customers", "sellers", "products", "geolocation",
    "orders", "order_items", "order_payments", "order_reviews"
]

@task(log_prints=True)
def silver_table(table_name: str):
    print(f"Procesando tabla: {table_name}")
    
    df = pd.read_sql(f"SELECT * FROM raw.{table_name}", engine_bronze)
    
    # Limpieza estándar
    df = df.drop_duplicates()
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # Tipado de fechas (según tabla)
    date_cols = {
        "orders": ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                   "order_delivered_customer_date", "order_estimated_delivery_date"],
        "order_reviews": ["review_creation_date", "review_answer_timestamp"]
    }
    
    if table_name in date_cols:
        for col in date_cols[table_name]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df.to_sql(table_name, engine_silver, schema="curated", if_exists="replace", index=False)
    print(f"Tabla {table_name} → {len(df):,} filas cargadas en silver.curated")

@flow(name="Bronze → Silver (Curated)")
def bronze_to_silver():
    for table in TABLES:
        silver_table(table)

if __name__ == "__main__":
    bronze_to_silver()