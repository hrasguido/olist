# src/flow.py
import prefect
from prefect import flow, task, get_run_logger
from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

@task(retries=2, retry_delay_seconds=10)
def extract() -> pd.DataFrame:
    logger = get_run_logger()
    file_path = "/workspace/data/olist_orders_dataset.csv"
    df = pd.read_csv(file_path)
    logger.info(f"Extraídos {len(df)} registros de {file_path}")
    return df

@task
def transform(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()

    date_cols = [
        'order_purchase_timestamp', 'order_approved_at',
        'order_delivered_carrier_date', 'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df = df[df['order_status'] == 'delivered'].copy()

    df['dias_entrega'] = (
        df['order_delivered_customer_date'] - df['order_purchase_timestamp']
    ).dt.days

    logger.info(f"Transformados: {len(df)} órdenes entregadas")
    return df

@task
def load(df: pd.DataFrame):
    logger = get_run_logger()

    engine = create_engine(
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/{os.getenv('POSTGRES_DB')}"
    )

    df.to_sql('orders', engine, if_exists='replace', index=False, method='multi')

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM orders")).scalar()

    logger.info(f"Cargadas {len(df)} filas → Total en BD: {count}")

@flow(name="ETL Olist → PostgreSQL")
def etl_olist():
    raw = extract()
    clean = transform(raw)
    load(clean)

if __name__ == "__main__":
    etl_olist()
