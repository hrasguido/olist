# src/gold_features.py
"""
Capa Gold - Tabla de Features Completa
Genera features temporales y logísticas para análisis de retrasos
"""
from prefect import flow, task, get_run_logger
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from typing import Dict, Tuple
from math import radians, sin, cos, sqrt, atan2

load_dotenv("/workspace/.env")

SILVER_CONN = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/silver"
GOLD_CONN = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@db:5432/gold"

engine_silver = create_engine(SILVER_CONN)
engine_gold = create_engine(GOLD_CONN)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia Haversine (en km) entre dos puntos geográficos."""
    R = 6371  # Radio de la Tierra en km
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def most_frequent(series):
    """Devuelve el primer valor de la moda de una serie."""
    mode_val = series.mode()
    return mode_val.iloc[0] if not mode_val.empty else np.nan


# ============================================================
# TASKS DE CARGA
# ============================================================

@task(log_prints=True)
def load_datasets_from_silver() -> Dict[str, pd.DataFrame]:
    """Carga todos los datasets necesarios desde Silver."""
    logger = get_run_logger()
    logger.info("📥 Cargando datasets desde Silver...")
    
    datasets = {}
    tables = {
        'orders': 'orders',
        'order_items': 'order_items',
        'customers': 'customers',
        'sellers': 'sellers',
        'products': 'products',
        'geolocation': 'geolocation',
        'order_payments': 'order_payments'
    }
    
    for key, table in tables.items():
        try:
            df = pd.read_sql(f"SELECT * FROM curated.{table}", engine_silver)
            datasets[key] = df
            logger.info(f"   ✅ {key}: {len(df):,} registros")
        except Exception as e:
            logger.warning(f"   ⚠️  Error cargando {table}: {str(e)}")
            datasets[key] = pd.DataFrame()
    
    return datasets


@task(log_prints=True)
def load_master_table() -> pd.DataFrame:
    """Carga la master table desde Gold."""
    logger = get_run_logger()
    logger.info("📥 Cargando master_table desde Gold...")
    
    df = pd.read_sql("SELECT * FROM dm.master_table", engine_gold)
    logger.info(f"   ✅ Cargados {len(df):,} registros")
    
    return df


# ============================================================
# GENERACIÓN DE FEATURES TEMPORALES
# ============================================================

@task(log_prints=True)
def generate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera variables de tiempo clave para el análisis de retrasos."""
    logger = get_run_logger()
    logger.info("🔨 Generando features temporales...")
    
    df = df.copy()
    
    # Convertir columnas de fecha
    logger.info("   📅 Convirtiendo columnas de fecha...")
    date_cols = [
        'order_purchase_timestamp', 
        'order_approved_at', 
        'order_delivered_carrier_date', 
        'order_estimated_delivery_date',
        'order_delivered_customer_date'
    ]
    
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Features de fecha de compra
    logger.info("   🛒 Generando features de fecha de compra...")
    if 'order_purchase_timestamp' in df.columns:
        df['purchase_day_of_week'] = df['order_purchase_timestamp'].dt.day_name()
        df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
        df['purchase_month'] = df['order_purchase_timestamp'].dt.month_name()
        df['is_weekend_purchase'] = df['order_purchase_timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
        df['purchase_quarter'] = df['order_purchase_timestamp'].dt.quarter
        
        hour = df['purchase_hour']
        df['is_night_purchase'] = ((hour >= 22) | (hour < 6)).astype(int)
        
        month = df['order_purchase_timestamp'].dt.month
        df['is_holiday_season'] = month.isin([11, 12]).astype(int)
    
    # Features de tiempo entre eventos
    logger.info("   ⏱️  Generando features de tiempo entre eventos...")
    
    if 'order_approved_at' in df.columns and 'order_purchase_timestamp' in df.columns:
        df['days_to_approve'] = (
            df['order_approved_at'] - df['order_purchase_timestamp']
        ).dt.total_seconds() / (60 * 60 * 24)
        df.loc[df['days_to_approve'] < 0, 'days_to_approve'] = np.nan
    
    if 'order_delivered_carrier_date' in df.columns and 'order_approved_at' in df.columns:
        df['days_to_carrier'] = (
            df['order_delivered_carrier_date'] - df['order_approved_at']
        ).dt.total_seconds() / (60 * 60 * 24)
        df.loc[df['days_to_carrier'] < 0, 'days_to_carrier'] = np.nan
    
    if 'order_estimated_delivery_date' in df.columns and 'order_purchase_timestamp' in df.columns:
        df['estimated_delivery_days'] = (
            df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
        ).dt.total_seconds() / (60 * 60 * 24)
        df.loc[df['estimated_delivery_days'] < 0, 'estimated_delivery_days'] = np.nan
    
    if 'order_estimated_delivery_date' in df.columns and 'order_delivered_carrier_date' in df.columns:
        df['days_in_transit_estimated'] = (
            df['order_estimated_delivery_date'] - df['order_delivered_carrier_date']
        ).dt.total_seconds() / (60 * 60 * 24)
        df.loc[df['days_in_transit_estimated'] < 0, 'days_in_transit_estimated'] = np.nan
    
    if 'order_delivered_customer_date' in df.columns and 'order_delivered_carrier_date' in df.columns:
        df['days_in_transit_actual'] = (
            df['order_delivered_customer_date'] - df['order_delivered_carrier_date']
        ).dt.total_seconds() / (60 * 60 * 24)
        df.loc[df['days_in_transit_actual'] < 0, 'days_in_transit_actual'] = np.nan
    
    logger.info(f"   ✅ Features temporales generadas")
    
    return df


# ============================================================
# GENERACIÓN DE FEATURES LOGÍSTICAS
# ============================================================

@task(log_prints=True)
def generate_logistics_features(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Genera features logísticas desde datasets de Silver."""
    logger = get_run_logger()
    logger.info("📦 Generando features logísticas...")
    
    orders_df = datasets['orders'].copy()
    order_items_df = datasets['order_items'].copy()
    products_df = datasets['products'].copy()
    sellers_df = datasets['sellers'].copy()
    customers_df = datasets['customers'].copy()
    geolocation_df = datasets['geolocation'].copy()
    
    # Normalizar IDs a string
    for df in [orders_df, customers_df, order_items_df, sellers_df, products_df]:
        for col in ['order_id', 'customer_id', 'seller_id', 'product_id']:
            if col in df.columns:
                df[col] = df[col].astype(str)
    
    # Normalizar columnas físicas de productos
    logger.info("   📏 Procesando dimensiones de productos...")
    product_physical_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
    for col in product_physical_cols:
        if col in products_df.columns:
            products_df[col] = pd.to_numeric(products_df[col], errors='coerce').fillna(0)
    
    # Calcular volumen
    if all(col in products_df.columns for col in ['product_length_cm', 'product_height_cm', 'product_width_cm']):
        products_df['product_volume_cm3'] = (
            products_df['product_length_cm'] * 
            products_df['product_height_cm'] * 
            products_df['product_width_cm']
        )
        products_df['product_volume_cm3'] = products_df['product_volume_cm3'].fillna(0)
    
    # Normalizar precios y flete
    if 'price' in order_items_df.columns:
        order_items_df['price'] = pd.to_numeric(order_items_df['price'], errors='coerce').fillna(0)
    if 'freight_value' in order_items_df.columns:
        order_items_df['freight_value'] = pd.to_numeric(order_items_df['freight_value'], errors='coerce').fillna(0)
    
    # Merge order_items con products
    merge_cols = ['product_id']
    if 'product_weight_g' in products_df.columns:
        merge_cols.append('product_weight_g')
    if 'product_volume_cm3' in products_df.columns:
        merge_cols.append('product_volume_cm3')
    if 'product_category_name' in products_df.columns:
        merge_cols.append('product_category_name')
    
    order_items_products = order_items_df.merge(
        products_df[merge_cols], 
        on='product_id', 
        how='left'
    )
    
    # Agregar por order_id
    logger.info("   📊 Agregando métricas por orden...")
    agg_dict = {
        'order_item_id': 'count',
        'seller_id': 'nunique'
    }
    
    if 'product_weight_g' in order_items_products.columns:
        agg_dict['product_weight_g'] = 'sum'
    if 'product_volume_cm3' in order_items_products.columns:
        agg_dict['product_volume_cm3'] = 'sum'
    if 'freight_value' in order_items_products.columns:
        agg_dict['freight_value'] = 'sum'
    if 'price' in order_items_products.columns:
        agg_dict['price'] = ['sum', 'mean']
    if 'product_category_name' in order_items_products.columns:
        agg_dict['product_category_name'] = ['nunique', most_frequent]
    
    logistics_features_df = order_items_products.groupby('order_id').agg(agg_dict).reset_index()
    
    # Aplanar columnas multi-nivel si existen
    if isinstance(logistics_features_df.columns, pd.MultiIndex):
        logistics_features_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                          for col in logistics_features_df.columns.values]
    
    # Renombrar columnas
    rename_map = {
        'order_item_id': 'order_item_count',
        'seller_id': 'num_sellers',
        'product_weight_g': 'total_weight',
        'product_volume_cm3': 'total_volume',
        'price_sum': 'total_price',
        'price_mean': 'avg_product_price',
        'product_category_name_nunique': 'num_unique_categories',
        'product_category_name_most_frequent': 'most_frequent_product_category'
    }
    logistics_features_df.rename(columns=rename_map, inplace=True)
    
    # Rellenar valores nulos en categoría
    if 'most_frequent_product_category' in logistics_features_df.columns:
        logistics_features_df['most_frequent_product_category'] = logistics_features_df['most_frequent_product_category'].fillna('unknown')
    
    # Agregar estados de seller y customer
    logger.info("   🗺️  Agregando información geográfica...")
    
    if 'seller_state' in sellers_df.columns:
        order_seller_state = order_items_df.sort_values('order_item_id').drop_duplicates('order_id').merge(
            sellers_df[['seller_id', 'seller_state']], 
            on='seller_id', 
            how='left'
        )[['order_id', 'seller_state']]
        
        logistics_features_df = logistics_features_df.merge(order_seller_state, on='order_id', how='left')
        logistics_features_df['seller_state'] = logistics_features_df['seller_state'].fillna('UNKNOWN')
    
    if 'customer_state' in customers_df.columns:
        order_customer_state = orders_df[['order_id', 'customer_id']].merge(
            customers_df[['customer_id', 'customer_state']], 
            on='customer_id', 
            how='left'
        )[['order_id', 'customer_state']]
        
        logistics_features_df = logistics_features_df.merge(order_customer_state, on='order_id', how='left')
        logistics_features_df['customer_state'] = logistics_features_df['customer_state'].fillna('UNKNOWN').astype(str)
    
    # Calcular distancia de envío
    logger.info("   📍 Calculando distancias de envío...")

    # Calcular same_state (cliente y vendedor en el mismo estado)
    if 'seller_state' in logistics_features_df.columns and 'customer_state' in logistics_features_df.columns:
        logistics_features_df['same_state'] = (
            logistics_features_df['customer_state'] == logistics_features_df['seller_state']
        ).astype(int)
        
        # Manejar casos donde alguno es UNKNOWN
        mask_unknown = (logistics_features_df['customer_state'] == 'UNKNOWN') | (logistics_features_df['seller_state'] == 'UNKNOWN')
        logistics_features_df.loc[mask_unknown, 'same_state'] = np.nan
        
        same_state_count = logistics_features_df['same_state'].sum()
        if logistics_features_df['same_state'].notna().sum() > 0:
            same_state_pct = (same_state_count / logistics_features_df['same_state'].notna().sum()) * 100
            logger.info(f"      • Órdenes mismo estado: {same_state_count:,.0f} ({same_state_pct:.1f}%)")
    
    if not geolocation_df.empty and all(col in geolocation_df.columns for col in ['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']):
        geo_df = geolocation_df[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']].copy()
        geo_df['geolocation_zip_code_prefix'] = pd.to_numeric(geo_df['geolocation_zip_code_prefix'], errors='coerce')
        geo_df_clean = geo_df.groupby('geolocation_zip_code_prefix').agg(
            lat=('geolocation_lat', 'mean'), 
            lng=('geolocation_lng', 'mean')
        ).reset_index()
        
        # Preparar zips de customer y seller
        customer_zip = customers_df[['customer_id', 'customer_zip_code_prefix']].merge(
            orders_df[['customer_id', 'order_id']], 
            on='customer_id', 
            how='right'
        )
        seller_zip = sellers_df[['seller_id', 'seller_zip_code_prefix']].merge(
            order_items_df[['seller_id', 'order_id']].drop_duplicates(), 
            on='seller_id', 
            how='right'
        )
        
        customer_zip['customer_zip_code_prefix'] = pd.to_numeric(customer_zip['customer_zip_code_prefix'], errors='coerce')
        seller_zip['seller_zip_code_prefix'] = pd.to_numeric(seller_zip['seller_zip_code_prefix'], errors='coerce')
        
        # Merge con coordenadas
        df_distance = customer_zip.merge(seller_zip, on='order_id', how='inner').merge(
            geo_df_clean.rename(columns={
                'geolocation_zip_code_prefix': 'customer_zip_code_prefix', 
                'lat': 'cust_lat', 
                'lng': 'cust_lng'
            }),
            on='customer_zip_code_prefix', 
            how='left'
        ).merge(
            geo_df_clean.rename(columns={
                'geolocation_zip_code_prefix': 'seller_zip_code_prefix', 
                'lat': 'seller_lat', 
                'lng': 'seller_lng'
            }),
            on='seller_zip_code_prefix', 
            how='left'
        )
        
        # Calcular distancia Haversine
        df_distance['shipping_distance_km'] = df_distance.apply(
            lambda row: haversine_distance(
                row['cust_lat'], row['cust_lng'], 
                row['seller_lat'], row['seller_lng']
            ) if all(pd.notna(row[['cust_lat', 'cust_lng', 'seller_lat', 'seller_lng']])) else np.nan,
            axis=1
        )
        
        logistics_features_df = logistics_features_df.merge(
            df_distance[['order_id', 'shipping_distance_km']], 
            on='order_id', 
            how='left'
        )
        logger.info(f"      • Distancias calculadas: {logistics_features_df['shipping_distance_km'].notna().sum():,}")
    
    # Features derivadas
    logger.info("   🧮 Calculando features derivadas...")
    
    if 'order_item_count' in logistics_features_df.columns:
        if 'freight_value' in logistics_features_df.columns:
            logistics_features_df['freight_per_item'] = np.where(
                logistics_features_df['order_item_count'] > 0,
                logistics_features_df['freight_value'] / logistics_features_df['order_item_count'],
                0
            )
        
        if 'total_weight' in logistics_features_df.columns:
            logistics_features_df['weight_per_item'] = np.where(
                logistics_features_df['order_item_count'] > 0,
                logistics_features_df['total_weight'] / logistics_features_df['order_item_count'],
                np.nan
            )
        
        if 'total_volume' in logistics_features_df.columns:
            logistics_features_df['volume_per_item'] = np.where(
                logistics_features_df['order_item_count'] > 0,
                logistics_features_df['total_volume'] / logistics_features_df['order_item_count'],
                np.nan
            )
    
    if 'num_sellers' in logistics_features_df.columns and 'order_item_count' in logistics_features_df.columns:
        logistics_features_df['items_per_seller'] = np.where(
            logistics_features_df['num_sellers'] > 0,
            logistics_features_df['order_item_count'] / logistics_features_df['num_sellers'],
            np.nan
        )
    
    if 'freight_value' in logistics_features_df.columns and 'total_price' in logistics_features_df.columns:
        logistics_features_df['freight_ratio'] = np.where(
            logistics_features_df['total_price'] > 0,
            logistics_features_df['freight_value'] / logistics_features_df['total_price'],
            np.nan
        )
    
    if 'freight_value' in logistics_features_df.columns and 'total_weight' in logistics_features_df.columns:
        logistics_features_df['avg_shipping_cost_per_kg'] = np.where(
            logistics_features_df['total_weight'] > 0,
            logistics_features_df['freight_value'] / logistics_features_df['total_weight'],
            np.nan
        )
    
    # Features binarias
    if 'shipping_distance_km' in logistics_features_df.columns:
        logistics_features_df['is_long_distance'] = (
            logistics_features_df['shipping_distance_km'] > 1000
        ).astype(int)
    
    if 'total_weight' in logistics_features_df.columns:
        weight_p75 = logistics_features_df['total_weight'].quantile(0.75)
        logistics_features_df['is_heavy_order'] = (
            logistics_features_df['total_weight'] > weight_p75
        ).astype(int)
        logger.info(f"      • Umbral peso pesado (P75): {weight_p75:.2f}g")
    
    # Features de producto
    if 'num_unique_categories' in logistics_features_df.columns:
        logistics_features_df['is_multi_category_order'] = (
            logistics_features_df['num_unique_categories'] > 1
        ).astype(int)
    
    if 'avg_product_price' in logistics_features_df.columns:
        price_p75 = logistics_features_df['avg_product_price'].quantile(0.75)
        logistics_features_df['is_expensive_order'] = (
            logistics_features_df['avg_product_price'] > price_p75
        ).astype(int)
        logger.info(f"      • Umbral precio caro (P75): {price_p75:.2f}")
    
    # Eliminar total_price (ya no se necesita)
    if 'total_price' in logistics_features_df.columns:
        logistics_features_df = logistics_features_df.drop(columns=['total_price'])
    
    logger.info(f"   ✅ Features logísticas generadas: {len(logistics_features_df):,} registros")
    
    return logistics_features_df

@task(log_prints=True)
def generate_payment_features(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Genera features de pago."""
    logger = get_run_logger()
    logger.info("💳 Generando features de pago...")
    
    orders_df = datasets['orders'].copy()
    order_payments_df = datasets.get('order_payments', pd.DataFrame())
    
    if order_payments_df.empty:
        logger.warning("   ⚠️  No hay datos de order_payments")
        return pd.DataFrame(columns=['order_id'])
    
    df = order_payments_df.copy()
    
    # Normalizar IDs y valores
    df['order_id'] = df['order_id'].astype(str)
    df['payment_value'] = pd.to_numeric(df['payment_value'], errors='coerce')
    df['payment_installments'] = pd.to_numeric(df['payment_installments'], errors='coerce').fillna(0).astype(int)
    
    # Feature binaria para boleto
    df['is_boleto'] = np.where(df['payment_type'] == 'boleto', 1, 0)
    
    # Agregar por order_id
    payment_features = df.groupby('order_id').agg(
        payment_value=('payment_value', 'sum'),
        payment_installments=('payment_installments', 'max'),
        payment_type=('payment_type', lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'),
        payment_delay_risk=('is_boleto', 'max'),
        num_payment_methods=('payment_type', 'nunique')
    ).reset_index()
    
    payment_features['payment_value'] = payment_features['payment_value'].fillna(0)
    
    # NUEVAS FEATURES
    logger.info("   💰 Calculando features derivadas de pago...")
    payment_features['has_multiple_payment_methods'] = (
        payment_features['num_payment_methods'] > 1
    ).astype(int)
    
    payment_features['installment_ratio'] = np.where(
        payment_features['payment_value'] > 0, 
        payment_features['payment_installments'] / payment_features['payment_value'], 
        np.nan
    )
    
    high_value_threshold = payment_features['payment_value'].quantile(0.9)
    payment_features['is_high_value_order'] = (
        payment_features['payment_value'] >= high_value_threshold
    ).astype(int)
    logger.info(f"      • Umbral alto valor (P90): {high_value_threshold:.2f}")
    
    logger.info(f"   ✅ Features de pago generadas: {len(payment_features):,} registros")
    
    return payment_features

@task(log_prints=True)
def generate_customer_features(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Genera features de cliente y targets."""
    logger = get_run_logger()
    logger.info("👥 Generando features de cliente...")
    
    orders_df = datasets['orders'].copy()
    customers_df = datasets['customers'].copy()
    
    # Normalizar IDs
    orders_df['customer_id'] = orders_df['customer_id'].astype(str)
    customers_df['customer_id'] = customers_df['customer_id'].astype(str)
    
    # Merge orders con customers
    df_merged = orders_df.merge(
        customers_df[['customer_id', 'customer_unique_id', 'customer_state', 'customer_city']], 
        on='customer_id', 
        how='left'
    )
    
    # Convertir fechas
    date_cols = ['order_delivered_customer_date', 'order_estimated_delivery_date', 'order_purchase_timestamp']
    for col in date_cols:
        if col in df_merged.columns:
            df_merged[col] = pd.to_datetime(df_merged[col], errors='coerce')
    
    # TARGETS
    logger.info("   🎯 Calculando targets de retraso...")
    if 'order_delivered_customer_date' in df_merged.columns and 'order_estimated_delivery_date' in df_merged.columns:
        df_merged['delay_days'] = (
            df_merged['order_delivered_customer_date'] - df_merged['order_estimated_delivery_date']
        ).dt.total_seconds() / (60 * 60 * 24)
        
        df_merged['is_late_delivery'] = (df_merged['delay_days'] > 0).astype(int)
        df_merged.loc[df_merged['delay_days'].isna(), 'is_late_delivery'] = np.nan
    
    # Features de historial de cliente
    logger.info("   📊 Calculando historial de cliente...")
    df_merged = df_merged.sort_values(by=['customer_unique_id', 'order_purchase_timestamp'])
    
    df_merged['num_previous_orders'] = df_merged.groupby('customer_unique_id').cumcount()
    
    # Calcular tasa de retraso acumulada
    if 'is_late_delivery' in df_merged.columns:
        df_merged['cumulative_delay_sum'] = df_merged.groupby('customer_unique_id')['is_late_delivery'].transform(
            lambda x: x.shift(1).cumsum()
        )
        df_merged['cumulative_order_count'] = df_merged.groupby('customer_unique_id')['is_late_delivery'].transform(
            lambda x: x.shift(1).notna().cumsum()
        )
        df_merged['avg_previous_delay_rate'] = (
            df_merged['cumulative_delay_sum'] / df_merged['cumulative_order_count']
        ).fillna(0)
    else:
        df_merged['avg_previous_delay_rate'] = 0
    
    # NUEVAS FEATURES
    logger.info("   🏆 Calculando nivel de lealtad...")
    conds = [
        df_merged['num_previous_orders'] == 0, 
        df_merged['num_previous_orders'].between(1, 3), 
        df_merged['num_previous_orders'] > 3
    ]
    choices = ['new', 'regular', 'loyal']
    df_merged['customer_loyalty_level'] = np.select(conds, choices, default='new')
    
    # Conteo de órdenes por ciudad
    city_counts = df_merged.groupby('customer_city')['order_id'].transform('count')
    df_merged['customer_city_order_count'] = city_counts
    
    # Seleccionar columnas finales
    customer_cols = [
        'order_id', 'customer_unique_id', 'customer_state', 'customer_city', 
        'num_previous_orders', 'avg_previous_delay_rate', 'customer_loyalty_level', 
        'customer_city_order_count'
    ]
    
    # Agregar targets si existen
    if 'delay_days' in df_merged.columns:
        customer_cols.extend(['delay_days', 'is_late_delivery'])
    
    customer_features = df_merged[[col for col in customer_cols if col in df_merged.columns]].copy()
    
    logger.info(f"   ✅ Features de cliente generadas: {len(customer_features):,} registros")
    
    return customer_features
# ============================================================
# COMBINACIÓN Y GUARDADO
# ============================================================

@task(log_prints=True)
def combine_features(
    temporal_df: pd.DataFrame, 
    logistics_df: pd.DataFrame,
    payment_df: pd.DataFrame,
    customer_df: pd.DataFrame
) -> pd.DataFrame:
    """Combina todas las features: temporales, logísticas, pago y cliente."""
    logger = get_run_logger()
    logger.info("🔗 Combinando todas las features...")
    
    # Seleccionar columnas temporales
    temporal_cols = [
        'order_id',
        'purchase_day_of_week',
        'purchase_hour',
        'purchase_month',
        'purchase_quarter',
        'is_weekend_purchase',
        'is_night_purchase',
        'is_holiday_season',
        'days_to_approve',
        'days_to_carrier',
        'estimated_delivery_days',
        'days_in_transit_estimated',
        'days_in_transit_actual'
    ]
    
    temporal_features = temporal_df[[col for col in temporal_cols if col in temporal_df.columns]].copy()
    
    # Merge con logistics
    logger.info("   📦 Agregando features logísticas...")
    combined_df = temporal_features.merge(logistics_df, on='order_id', how='left')
    
    # Merge con payment
    if not payment_df.empty:
        logger.info("   💳 Agregando features de pago...")
        combined_df = combined_df.merge(payment_df, on='order_id', how='left')
    
    # Merge con customer
    if not customer_df.empty:
        logger.info("   👥 Agregando features de cliente...")
        combined_df = combined_df.merge(customer_df, on='order_id', how='left')
    
    # Features derivadas finales (combinaciones entre grupos)
    logger.info("   🧮 Calculando features derivadas finales...")
    
    # Distancia por día estimado
    if 'shipping_distance_km' in combined_df.columns and 'estimated_delivery_days' in combined_df.columns:
        combined_df['distance_per_estimated_day'] = np.where(
            combined_df['estimated_delivery_days'] > 0, 
            combined_df['shipping_distance_km'] / combined_df['estimated_delivery_days'], 
            np.nan
        )
    
    # Items por día estimado
    if 'order_item_count' in combined_df.columns and 'estimated_delivery_days' in combined_df.columns:
        combined_df['items_per_estimated_day'] = np.where(
            combined_df['estimated_delivery_days'] > 0, 
            combined_df['order_item_count'] / combined_df['estimated_delivery_days'], 
            np.nan
        )
    
    # Flete por día estimado
    if 'freight_value' in combined_df.columns and 'estimated_delivery_days' in combined_df.columns:
        combined_df['freight_per_estimated_day'] = np.where(
            combined_df['estimated_delivery_days'] > 0, 
            combined_df['freight_value'] / combined_df['estimated_delivery_days'], 
            np.nan
        )
    
    # Precio por peso
    if 'total_weight' in combined_df.columns and 'payment_value' in combined_df.columns:
        combined_df['price_per_weight'] = np.where(
            combined_df['total_weight'] > 0, 
            combined_df['payment_value'] / combined_df['total_weight'], 
            np.nan
        )
    
    # Feature binaria: es envío interestatal
    if 'same_state' in combined_df.columns:
        combined_df['is_interstate'] = (combined_df['same_state'] == 0).astype(int)
        # Manejar NaN en same_state
        combined_df.loc[combined_df['same_state'].isna(), 'is_interstate'] = np.nan
    
        logger.info(f"   ✅ Features combinadas: {len(combined_df):,} registros, {len(combined_df.columns)} columnas")
    
    # ============================================================
    # VERIFICAR Y ELIMINAR DUPLICADOS POR ORDER_ID
    # ============================================================
    logger.info("   🧹 Verificando duplicados...")
    initial_count = len(combined_df)
    
    if 'order_id' in combined_df.columns:
        duplicates_count = combined_df.duplicated(subset=['order_id'], keep='first').sum()
        
        if duplicates_count > 0:
            logger.warning(f"      ⚠️  Duplicados detectados: {duplicates_count:,} registros")
            logger.info(f"      • Causa: Merges múltiples entre features")
            
            # Eliminar duplicados manteniendo el primer registro
            combined_df = combined_df.drop_duplicates(subset=['order_id'], keep='first')
            
            logger.info(f"      ✅ Duplicados eliminados: {initial_count - len(combined_df):,}")
            logger.info(f"      📊 Registros únicos finales: {len(combined_df):,}")
        else:
            logger.info(f"      ✅ No se detectaron duplicados en order_id")
    
    return combined_df


@task(log_prints=True)
def save_features_table(features_df: pd.DataFrame) -> Dict:
    """Guarda la tabla de features en Gold."""
    logger = get_run_logger()
    logger.info("💾 Guardando tabla de features en Gold...")
    
    # Guardar en gold.dm.features
    features_df.to_sql(
        'features',
        engine_gold,
        schema='dm',
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=5000
    )
    
    logger.info(f"   ✅ Tabla 'gold.dm.features' creada")
    logger.info(f"   📊 {len(features_df):,} registros, {len(features_df.columns)} columnas")
    
    # Estadísticas de valores nulos
    null_counts = features_df.isnull().sum()
    if null_counts.sum() > 0:
        logger.info("   ⚠️  Valores nulos por columna:")
        for col, count in null_counts[null_counts > 0].items():
            pct = (count / len(features_df)) * 100
            logger.info(f"      • {col}: {count:,} ({pct:.1f}%)")
    
    return {
        "status": "success",
        "total_rows": len(features_df),
        "total_columns": len(features_df.columns),
        "null_values": int(null_counts.sum())
    }


# ============================================================
# FLUJO PRINCIPAL
# ============================================================

@flow(name="Gold → Features Table (Completa)", log_prints=True)
def generate_features_table():
    """
    Flujo que genera la tabla de features completa
    (temporales + logísticas + pago + cliente) desde Silver y Gold
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("🥇 INICIANDO GENERACIÓN DE FEATURES COMPLETAS - GOLD")
    logger.info("=" * 80)
    
    # 1. Cargar master table para features temporales
    master_df = load_master_table()
    
    # 2. Generar features temporales
    temporal_df = generate_temporal_features(master_df)
    
    # 3. Cargar datasets desde Silver
    datasets = load_datasets_from_silver()
    
    # 4. Generar features logísticas
    logistics_df = generate_logistics_features(datasets)
    
    # 5. Generar features de pago
    payment_df = generate_payment_features(datasets)
    
    # 6. Generar features de cliente
    customer_df = generate_customer_features(datasets)
    
    # 7. Combinar todas las features
    features_df = combine_features(temporal_df, logistics_df, payment_df, customer_df)
    
    # 8. Guardar en Gold
    result = save_features_table(features_df)
    
    logger.info("=" * 80)
    logger.info("✅ TABLA DE FEATURES COMPLETADA")
    logger.info("=" * 80)
    logger.info(f"📊 Resumen:")
    logger.info(f"   - Registros: {result['total_rows']:,}")
    logger.info(f"   - Columnas: {result['total_columns']}")
    logger.info(f"   - Valores nulos: {result['null_values']:,}")
    logger.info("=" * 80)
    
    return result


if __name__ == "__main__":
    generate_features_table()