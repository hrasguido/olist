# src/etl_gold.py
from prefect import flow, task
from sqlalchemy import create_engine
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv("/workspace/.env")

SILVER_CONN = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/silver"
GOLD_CONN   = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/gold"

engine_silver = create_engine(SILVER_CONN)
engine_gold   = create_engine(GOLD_CONN)

@task(log_prints=True)
def build_fact_sales():
    query = """
    WITH base AS (
        SELECT 
            op.payment_value,
            oi.price + oi.freight_value AS total_order_value,
            COALESCE(p.product_length_cm, 0) * COALESCE(p.product_height_cm, 0) * COALESCE(p.product_width_cm, 0) AS product_volume_cm3,
            oi.price AS avg_price_per_item,
            COALESCE(p.product_weight_g, 0) AS product_weight_g,
            c.customer_zip_code_prefix,
            s.seller_zip_code_prefix,
            EXTRACT(MONTH FROM o.order_purchase_timestamp) AS purchase_month,
            EXTRACT(YEAR FROM o.order_purchase_timestamp) AS purchase_year,
            (c.customer_state = s.seller_state) AS same_state,
            EXTRACT(HOUR FROM (o.order_approved_at - o.order_purchase_timestamp)) AS approval_time_hours,
            EXTRACT(DAY FROM (o.order_estimated_delivery_date - o.order_delivered_customer_date)) AS estimated_time_days,
            EXTRACT(DAY FROM (o.order_delivered_customer_date - o.order_delivered_carrier_date)) AS carrier_time_days,
            EXTRACT(DAY FROM (o.order_delivered_customer_date - o.order_purchase_timestamp)) AS delivery_time_days,
            EXTRACT(DAY FROM (o.order_delivered_customer_date - o.order_estimated_delivery_date)) AS delivery_diff_days,
            oi.freight_value,
            -- Campos geográficos (simulados por ahora, luego puedes enriquecer)
            0.0 AS customer_lat,
            0.0 AS customer_lng,
            0.0 AS seller_lat,
            0.0 AS seller_lng,
            0.0 AS distance_km
        FROM curated.order_items oi
        JOIN curated.orders o ON oi.order_id = o.order_id
        JOIN curated.customers c ON o.customer_id = c.customer_id
        JOIN curated.products p ON oi.product_id = p.product_id
        JOIN curated.sellers s ON oi.seller_id = s.seller_id
        JOIN curated.order_payments op ON oi.order_id = op.order_id
        WHERE o.order_status = 'delivered'
    )
    SELECT * FROM base
    """
    
    df = pd.read_sql(query, engine_silver)
    df.to_sql("fact_sales", engine_gold, schema="dm", if_exists="replace", index=False)
    print(f"fact_sales cargada con {len(df):,} registros y 20 campos objetivo")

@flow(name="Silver → Gold (Modelo Dimensional)")
def silver_to_gold():
    build_fact_sales()

if __name__ == "__main__":
    silver_to_gold()