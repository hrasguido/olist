import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from typing import Dict
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def generate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera variables de tiempo clave para el análisis de retrasos."""
    df = df.copy()

    date_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_estimated_delivery_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['purchase_day_of_week'] = df['order_purchase_timestamp'].dt.day_name(locale='en_US.UTF-8')
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    
    df['days_to_approve'] = (df['order_approved_at'] - df['order_purchase_timestamp']).dt.total_seconds() / (60 * 60 * 24)
    df.loc[df['days_to_approve'] < 0, 'days_to_approve'] = np.nan
    
    df['days_to_carrier'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.total_seconds() / (60 * 60 * 24)
    df.loc[df['days_to_carrier'] < 0, 'days_to_carrier'] = np.nan
    
    df['estimated_delivery_days'] = (df['order_estimated_delivery_date'] - df['order_purchase_timestamp']).dt.total_seconds() / (60 * 60 * 24)
    df.loc[df['estimated_delivery_days'] < 0, 'estimated_delivery_days'] = np.nan
    
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month_name(locale='en_US.UTF-8')
    
    df['days_in_transit_estimated'] = (df['order_estimated_delivery_date'] - df['order_delivered_carrier_date']).dt.total_seconds() / (60 * 60 * 24)
    df.loc[df['days_in_transit_estimated'] < 0, 'days_in_transit_estimated'] = np.nan
    
    df['days_in_transit_actual'] = (df['order_delivered_customer_date'] - df['order_delivered_carrier_date']).dt.total_seconds() / (60 * 60 * 24)
    df.loc[df['days_in_transit_actual'] < 0, 'days_in_transit_actual'] = np.nan
    
    # NUEVAS FEATURES
    df['is_weekend_purchase'] = df['order_purchase_timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    df['purchase_quarter'] = df['order_purchase_timestamp'].dt.quarter
    hour = df['purchase_hour']
    df['is_night_purchase'] = ((hour >= 22) | (hour < 6)).astype(int)
    month = df['order_purchase_timestamp'].dt.month
    df['is_holiday_season'] = month.isin([11, 12]).astype(int)

    features = df[['order_id', 'purchase_day_of_week', 'purchase_hour', 'days_to_approve', 'days_to_carrier',
                   'estimated_delivery_days', 'purchase_month', 'days_in_transit_estimated', 'days_in_transit_actual',
                   'is_weekend_purchase', 'purchase_quarter', 'is_night_purchase', 'is_holiday_season']].copy()

    return features


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia Haversine (en km) entre dos puntos geográficos."""
    R = 6371
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def generate_logistics_features(orders_df, order_items_df, products_df, sellers_df, customers_df, geolocation_df):
    """Genera features logísticas."""
    for df in [orders_df, customers_df, order_items_df, sellers_df, products_df]:
        for col in ['order_id', 'customer_id', 'seller_id', 'product_id']:
            if col in df.columns:
                df[col] = df[col].astype(str)

    product_physical_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
    for col in product_physical_cols:
        products_df[col] = pd.to_numeric(products_df[col], errors='coerce').fillna(0)

    products_df['product_volume_cm3'] = products_df['product_length_cm'] * products_df['product_height_cm'] * products_df['product_width_cm']
    products_df['product_volume_cm3'] = products_df['product_volume_cm3'].fillna(0)

    order_items_df['price'] = pd.to_numeric(order_items_df['price'], errors='coerce').fillna(0)
    order_items_df['freight_value'] = pd.to_numeric(order_items_df['freight_value'], errors='coerce').fillna(0)

    order_items_products = order_items_df.merge(products_df[['product_id', 'product_weight_g', 'product_volume_cm3']], on='product_id', how='left')

    logistics_features_df = order_items_products.groupby('order_id').agg(
        order_item_count=('order_item_id', 'count'),
        num_sellers=('seller_id', 'nunique'),
        total_weight=('product_weight_g', 'sum'),
        total_volume=('product_volume_cm3', 'sum'),
        freight_value=('freight_value', 'sum'),
        total_price=('price', 'sum')
    ).reset_index()

    order_seller_state = order_items_df.sort_values('order_item_id').drop_duplicates('order_id').merge(
        sellers_df[['seller_id', 'seller_state']], on='seller_id', how='left')[['order_id', 'seller_state']]

    order_customer_state = orders_df[['order_id', 'customer_id']].merge(
        customers_df[['customer_id', 'customer_state']], on='customer_id', how='left')[['order_id', 'customer_state']]
    
    logistics_features_df = logistics_features_df.merge(order_seller_state, on='order_id', how='left')
    logistics_features_df = logistics_features_df.merge(order_customer_state, on='order_id', how='left')

    logistics_features_df['seller_state'] = logistics_features_df['seller_state'].fillna('UNKNOWN')
    logistics_features_df['customer_state'] = logistics_features_df['customer_state'].fillna('UNKNOWN').astype(str)
    
    geo_df = geolocation_df[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']].copy()
    geo_df['geolocation_zip_code_prefix'] = pd.to_numeric(geo_df['geolocation_zip_code_prefix'], errors='coerce')
    geo_df_clean = geo_df.groupby('geolocation_zip_code_prefix').agg(lat=('geolocation_lat', 'mean'), lng=('geolocation_lng', 'mean')).reset_index()

    customer_zip = customers_df[['customer_id', 'customer_zip_code_prefix']].merge(orders_df[['customer_id', 'order_id']], on='customer_id', how='right')
    seller_zip = sellers_df[['seller_id', 'seller_zip_code_prefix']].merge(order_items_df[['seller_id', 'order_id']].drop_duplicates(), on='seller_id', how='right')

    customer_zip['customer_zip_code_prefix'] = pd.to_numeric(customer_zip['customer_zip_code_prefix'], errors='coerce')
    seller_zip['seller_zip_code_prefix'] = pd.to_numeric(seller_zip['seller_zip_code_prefix'], errors='coerce')

    df_distance = customer_zip.merge(seller_zip, on='order_id', how='inner').merge(
        geo_df_clean.rename(columns={'geolocation_zip_code_prefix': 'customer_zip_code_prefix', 'lat': 'cust_lat', 'lng': 'cust_lng'}),
        on='customer_zip_code_prefix', how='left'
    ).merge(
        geo_df_clean.rename(columns={'geolocation_zip_code_prefix': 'seller_zip_code_prefix', 'lat': 'seller_lat', 'lng': 'seller_lng'}),
        on='seller_zip_code_prefix', how='left'
    )

    df_distance['shipping_distance_km'] = df_distance.apply(
        lambda row: haversine_distance(row['cust_lat'], row['cust_lng'], row['seller_lat'], row['seller_lng'])
        if all(pd.notna(row[['cust_lat', 'cust_lng', 'seller_lat', 'seller_lng']])) else np.nan,
        axis=1
    )

    logistics_features_df = logistics_features_df.merge(df_distance[['order_id', 'shipping_distance_km']], on='order_id', how='left')

    logistics_features_df['freight_per_item'] = np.where(logistics_features_df['order_item_count'] > 0, logistics_features_df['freight_value'] / logistics_features_df['order_item_count'], 0)
    logistics_features_df['freight_ratio'] = np.where(logistics_features_df['total_price'] > 0, logistics_features_df['freight_value'] / logistics_features_df['total_price'], np.nan)
    logistics_features_df['avg_shipping_cost_per_kg'] = np.where(logistics_features_df['total_weight'] > 0, logistics_features_df['freight_value'] / logistics_features_df['total_weight'], np.nan)
    
    # NUEVAS FEATURES
    logistics_features_df['weight_per_item'] = np.where(logistics_features_df['order_item_count'] > 0, logistics_features_df['total_weight'] / logistics_features_df['order_item_count'], np.nan)
    logistics_features_df['volume_per_item'] = np.where(logistics_features_df['order_item_count'] > 0, logistics_features_df['total_volume'] / logistics_features_df['order_item_count'], np.nan)
    logistics_features_df['items_per_seller'] = np.where(logistics_features_df['num_sellers'] > 0, logistics_features_df['order_item_count'] / logistics_features_df['num_sellers'], np.nan)
    logistics_features_df['is_long_distance'] = (logistics_features_df['shipping_distance_km'] > 1000).astype(int)
    weight_p75 = logistics_features_df['total_weight'].quantile(0.75)
    logistics_features_df['is_heavy_order'] = (logistics_features_df['total_weight'] > weight_p75).astype(int)
    
    logistics_features_df = logistics_features_df.drop(columns=['total_price'])

    return logistics_features_df


def most_frequent(series):
    """Devuelve el primer valor de la moda de una serie."""
    mode_val = series.mode()
    return mode_val.iloc[0] if not mode_val.empty else np.nan


def generate_product_features(order_items_df, products_df):
    """Genera features de producto."""
    order_items_df['product_id'] = order_items_df['product_id'].astype(str)
    products_df['product_id'] = products_df['product_id'].astype(str)

    order_items_products = order_items_df.merge(products_df[['product_id', 'product_category_name']], on='product_id', how='left')
    order_items_products['price'] = pd.to_numeric(order_items_products['price'], errors='coerce')

    product_features = order_items_products.groupby('order_id').agg(
        total_price=('price', 'sum'),
        avg_product_price=('price', 'mean'),
        num_unique_categories=('product_category_name', 'nunique'),
        most_frequent_product_category=('product_category_name', most_frequent)
    ).reset_index()

    product_features['most_frequent_product_category'] = product_features['most_frequent_product_category'].fillna('unknown')
    
    # NUEVAS FEATURES
    product_features['is_multi_category_order'] = (product_features['num_unique_categories'] > 1).astype(int)
    price_p75 = product_features['avg_product_price'].quantile(0.75)
    product_features['is_expensive_order'] = (product_features['avg_product_price'] > price_p75).astype(int)

    return product_features


def generate_payment_features(order_payments_df):
    """Genera features de pago."""
    df = order_payments_df.copy()

    df['order_id'] = df['order_id'].astype(str)
    df['payment_value'] = pd.to_numeric(df['payment_value'], errors='coerce')
    df['payment_installments'] = pd.to_numeric(df['payment_installments'], errors='coerce').fillna(0).astype(int)

    df['is_boleto'] = np.where(df['payment_type'] == 'boleto', 1, 0)

    payment_features = df.groupby('order_id').agg(
        payment_value=('payment_value', 'sum'),
        payment_installments=('payment_installments', 'max'),
        payment_type=('payment_type', lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'),
        payment_delay_risk=('is_boleto', 'max'),
        num_payment_methods=('payment_type', 'nunique')
    ).reset_index()

    payment_features['payment_value'] = payment_features['payment_value'].fillna(0)
    
    # NUEVAS FEATURES
    payment_features['has_multiple_payment_methods'] = (payment_features['num_payment_methods'] > 1).astype(int)
    payment_features['installment_ratio'] = np.where(payment_features['payment_value'] > 0, payment_features['payment_installments'] / payment_features['payment_value'], np.nan)
    high_value_threshold = payment_features['payment_value'].quantile(0.9)
    payment_features['is_high_value_order'] = (payment_features['payment_value'] >= high_value_threshold).astype(int)

    return payment_features


def generate_customer_features(orders_df, customers_df):
    """Genera features de cliente y targets."""
    orders_df['customer_id'] = orders_df['customer_id'].astype(str)
    customers_df['customer_id'] = customers_df['customer_id'].astype(str)

    df_merged = orders_df.merge(customers_df[['customer_id', 'customer_unique_id', 'customer_state', 'customer_city']], on='customer_id', how='left')

    date_cols = ['order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_cols:
        df_merged[col] = pd.to_datetime(df_merged[col], errors='coerce')

    # TARGETS
    df_merged['delay_days'] = (df_merged['order_delivered_customer_date'] - df_merged['order_estimated_delivery_date']).dt.total_seconds() / (60 * 60 * 24)
    df_merged['is_late_delivery'] = (df_merged['delay_days'] > 0).astype(int)
    df_merged.loc[df_merged['delay_days'].isna(), 'is_late_delivery'] = np.nan

    df_merged['purchase_timestamp'] = pd.to_datetime(df_merged['order_purchase_timestamp'], errors='coerce')
    df_merged = df_merged.sort_values(by=['customer_unique_id', 'purchase_timestamp'])

    df_merged['num_previous_orders'] = df_merged.groupby('customer_unique_id').cumcount()
    df_merged['cumulative_delay_sum'] = df_merged.groupby('customer_unique_id')['is_late_delivery'].transform(lambda x: x.shift(1).cumsum())
    df_merged['cumulative_order_count'] = df_merged.groupby('customer_unique_id')['is_late_delivery'].transform(lambda x: x.shift(1).notna().cumsum())
    df_merged['avg_previous_delay_rate'] = (df_merged['cumulative_delay_sum'] / df_merged['cumulative_order_count']).fillna(0)

    # NUEVAS FEATURES
    conds = [df_merged['num_previous_orders'] == 0, df_merged['num_previous_orders'].between(1, 3), df_merged['num_previous_orders'] > 3]
    choices = ['new', 'regular', 'loyal']
    df_merged['customer_loyalty_level'] = np.select(conds, choices, default='new')
    city_counts = df_merged.groupby('customer_city')['order_id'].transform('count')
    df_merged['customer_city_order_count'] = city_counts

    customer_features = df_merged[['order_id', 'customer_unique_id', 'customer_state', 'customer_city', 'num_previous_orders',
                                    'avg_previous_delay_rate', 'customer_loyalty_level', 'customer_city_order_count', 'delay_days', 'is_late_delivery']].copy()

    return customer_features


def generate_geo_features(orders_df, customers_df, sellers_df, order_items_df, geolocation_df):
    """Genera features geográficas."""
    orders_df['order_id'] = orders_df['order_id'].astype(str)

    geo_df = geolocation_df[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']].copy()
    geo_df['geolocation_zip_code_prefix'] = pd.to_numeric(geo_df['geolocation_zip_code_prefix'], errors='coerce')
    geo_centroids = geo_df.groupby('geolocation_zip_code_prefix').agg(lat=('geolocation_lat', 'mean'), lng=('geolocation_lng', 'mean')).reset_index()

    customer_loc = customers_df[['customer_id', 'customer_zip_code_prefix', 'customer_state']].merge(orders_df[['customer_id', 'order_id']], on='customer_id', how='right')
    customer_loc['customer_zip_code_prefix'] = pd.to_numeric(customer_loc['customer_zip_code_prefix'], errors='coerce')

    seller_loc = sellers_df[['seller_id', 'seller_zip_code_prefix', 'seller_state']].merge(order_items_df[['seller_id', 'order_id']].drop_duplicates(subset='order_id'), on='seller_id', how='right')
    seller_loc['seller_zip_code_prefix'] = pd.to_numeric(seller_loc['seller_zip_code_prefix'], errors='coerce')

    df_merged = customer_loc.merge(seller_loc[['order_id', 'seller_zip_code_prefix', 'seller_state']], on='order_id', how='inner')
    df_merged = df_merged.merge(geo_centroids.rename(columns={'geolocation_zip_code_prefix': 'customer_zip_code_prefix', 'lat': 'cust_lat', 'lng': 'cust_lng'}), on='customer_zip_code_prefix', how='left')
    df_merged = df_merged.merge(geo_centroids.rename(columns={'geolocation_zip_code_prefix': 'seller_zip_code_prefix', 'lat': 'seller_lat', 'lng': 'seller_lng'}), on='seller_zip_code_prefix', how='left')

    df_merged['same_state'] = (df_merged['customer_state'] == df_merged['seller_state']).astype(int)

    return df_merged[['order_id', 'same_state']].copy()


def build_features_and_targets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Función consolidada que genera todas las features y targets para clasificación y regresión."""
    
    print("Generando features temporales...")
    temporal_features = generate_temporal_features(datasets['orders'])
    
    print("Generando features logísticas...")
    logistics_features = generate_logistics_features(datasets['orders'], datasets['order_items'], datasets['products'], datasets['sellers'], datasets['customers'], datasets['geolocation'])
    
    print("Generando features de producto...")
    product_features = generate_product_features(datasets['order_items'], datasets['products'])
    
    print("Generando features de pago...")
    payment_features = generate_payment_features(datasets['order_payments'])
    
    print("Generando features de cliente y targets...")
    customer_features = generate_customer_features(datasets['orders'], datasets['customers'])
    
    print("Generando features geográficas...")
    geo_features = generate_geo_features(datasets['orders'], datasets['customers'], datasets['sellers'], datasets['order_items'], datasets['geolocation'])
    
    print("Uniendo todas las features por order_id...")
    df_final = customer_features.copy()
    df_final = df_final.merge(temporal_features, on='order_id', how='left')
    df_final = df_final.merge(logistics_features, on='order_id', how='left')
    df_final = df_final.merge(product_features, on='order_id', how='left')
    df_final = df_final.merge(payment_features, on='order_id', how='left')
    df_final = df_final.merge(geo_features, on='order_id', how='left')
    
    # Features derivadas finales (combinaciones entre grupos)
    df_final['distance_per_estimated_day'] = np.where(df_final['estimated_delivery_days'] > 0, df_final['shipping_distance_km'] / df_final['estimated_delivery_days'], np.nan)
    df_final['items_per_estimated_day'] = np.where(df_final['estimated_delivery_days'] > 0, df_final['order_item_count'] / df_final['estimated_delivery_days'], np.nan)
    df_final['freight_per_estimated_day'] = np.where(df_final['estimated_delivery_days'] > 0, df_final['freight_value'] / df_final['estimated_delivery_days'], np.nan)
    df_final['price_per_weight'] = np.where(df_final['total_weight'] > 0, df_final['total_price'] / df_final['total_weight'], np.nan)
    df_final['is_interstate'] = (df_final['same_state'] == 0).astype(int)
    
    print(f"Dataset final generado con {len(df_final)} filas y {len(df_final.columns)} columnas.")
    
    return df_final

def create_master_df(initial_master_df, features_and_targets_df)-> pd.DataFrame:
    df = initial_master_df.copy()
    date_cols_to_drop = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 
                         'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date',
                         'customer_id', 'customer_unique_id', 'customer_city', 'product_id', 'seller_id',
                         'seller_city', 'review_id', 'review_creation_date', 'review_answer_timestamp',
                         'product_category_name']
    
    df = df.drop(columns=[col for col in date_cols_to_drop if col in df.columns])

    # Merge con features_and_targets_df por order_id
    print(f"\nRealizando merge con features_and_targets_df...")
    
    master_df = df.merge(features_and_targets_df, on='order_id', how='inner')

    return master_df