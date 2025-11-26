"""
Módulo para cálculo de KPIs de negocio.
Analiza métricas operacionales derivadas del análisis de entregas.
"""

from prefect import task, get_run_logger
import pandas as pd
import numpy as np
import os
from typing import Dict


# Configurar carpeta de outputs (ruta relativa)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


@task(log_prints=True)
def calculate_business_kpis(master_df: pd.DataFrame) -> Dict:
    """
    Calcula y muestra KPIs de negocio derivados del análisis.
    
    Args:
        master_df: DataFrame con features, targets y predicciones
    
    Returns:
        dict con todos los KPIs calculados
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("📊 CALCULANDO KPIs DE NEGOCIO")
    logger.info("=" * 80)
    
    kpis = {}
    
    # ============================================================
    # KPIs OPERACIONALES GENERALES
    # ============================================================
    logger.info("")
    logger.info("📊 KPIs OPERACIONALES GENERALES")
    logger.info("-" * 80)
    
    total_orders = len(master_df)
    
    # Calcular entregas tardías (Delayed_time > 0)
    late_orders = (master_df['Delayed_time'] > 0).sum()
    on_time_orders = total_orders - late_orders
    late_rate = (late_orders / total_orders) * 100
    on_time_rate = 100 - late_rate
    
    kpis['total_orders'] = int(total_orders)
    kpis['late_orders'] = int(late_orders)
    kpis['on_time_orders'] = int(on_time_orders)
    kpis['late_rate'] = float(late_rate)
    kpis['on_time_rate'] = float(on_time_rate)
    
    logger.info(f"   Total de órdenes analizadas: {total_orders:,}")
    logger.info(f"   Entregas tardías: {late_orders:,} ({late_rate:.2f}%)")
    logger.info(f"   Entregas a tiempo: {on_time_orders:,} ({on_time_rate:.2f}%)")
    
    # Crear columna auxiliar para análisis posteriores
    master_df['is_late'] = (master_df['Delayed_time'] > 0).astype(int)
    
    # ============================================================
    # KPIs DE RETRASO
    # ============================================================
    logger.info("")
    logger.info("⏱️  KPIs DE RETRASO")
    logger.info("-" * 80)
    
    late_deliveries = master_df[master_df['Delayed_time'] > 0]
    if len(late_deliveries) > 0:
        avg_delay = late_deliveries['Delayed_time'].mean()
        median_delay = late_deliveries['Delayed_time'].median()
        max_delay = late_deliveries['Delayed_time'].max()
        min_delay = late_deliveries['Delayed_time'].min()
        std_delay = late_deliveries['Delayed_time'].std()
        
        kpis['avg_delay_days'] = float(avg_delay)
        kpis['median_delay_days'] = float(median_delay)
        kpis['max_delay_days'] = float(max_delay)
        kpis['min_delay_days'] = float(min_delay)
        kpis['std_delay_days'] = float(std_delay)
        
        logger.info(f"   Días promedio de retraso: {avg_delay:.2f} días")
        logger.info(f"   Mediana de retraso: {median_delay:.2f} días")
        logger.info(f"   Retraso máximo: {max_delay:.2f} días")
        logger.info(f"   Retraso mínimo: {min_delay:.2f} días")
        logger.info(f"   Desviación estándar: {std_delay:.2f} días")
    
    # ============================================================
    # KPIs DE PRECISIÓN DEL MODELO
    # ============================================================
    if 'Delayed_time_predicted' in master_df.columns:
        logger.info("")
        logger.info("🎯 KPIs DE PRECISIÓN DEL MODELO")
        logger.info("-" * 80)
        
        mae = master_df['prediction_error_abs'].mean()
        rmse = np.sqrt((master_df['prediction_error'] ** 2).mean())
        
        # Precisión en clasificación (tardío vs a tiempo)
        actual_late = (master_df['Delayed_time'] > 0).astype(int)
        predicted_late = (master_df['Delayed_time_predicted'] > 0).astype(int)
        accuracy = (actual_late == predicted_late).mean() * 100
        
        kpis['model_mae'] = float(mae)
        kpis['model_rmse'] = float(rmse)
        kpis['model_accuracy'] = float(accuracy)
        
        logger.info(f"   MAE (Error Absoluto Medio): {mae:.2f} días")
        logger.info(f"   RMSE: {rmse:.2f} días")
        logger.info(f"   Precisión en clasificación: {accuracy:.2f}%")
    
    # ============================================================
    # KPIs POR ESTADO DEL CLIENTE
    # ============================================================
    if 'customer_state' in master_df.columns:
        logger.info("")
        logger.info("🗺️  KPIs POR ESTADO DEL CLIENTE (Top 10)")
        logger.info("-" * 80)
        
        # ELIMINAR ESTA LÍNEA (ya se creó arriba)
        # master_df['is_late'] = (master_df['Delayed_time'] > 0).astype(int)
        
        state_kpis = master_df.groupby('customer_state').agg({
            'is_late': ['count', 'sum', 'mean'],
            'Delayed_time': 'mean'
        }).round(2)
        
        state_kpis.columns = ['total_orders', 'late_orders', 'late_rate', 'avg_delay']
        state_kpis = state_kpis.sort_values('late_rate', ascending=False).head(10)
        state_kpis['late_rate'] = state_kpis['late_rate'] * 100
        
        logger.info(f"\n{state_kpis.to_string()}")
        kpis['state_kpis'] = state_kpis
    
    # ============================================================
    # KPIs POR MÉTODO DE PAGO
    # ============================================================
    if 'payment_type' in master_df.columns:
        logger.info("")
        logger.info("💳 KPIs POR MÉTODO DE PAGO")
        logger.info("-" * 80)
        
        payment_kpis = master_df.groupby('payment_type').agg({
            'is_late': ['count', 'sum', 'mean'],
            'Delayed_time': 'mean'
        }).round(2)
        
        payment_kpis.columns = ['total_orders', 'late_orders', 'late_rate', 'avg_delay']
        payment_kpis = payment_kpis.sort_values('late_rate', ascending=False)
        payment_kpis['late_rate'] = payment_kpis['late_rate'] * 100
        
        logger.info(f"\n{payment_kpis.to_string()}")
        kpis['payment_kpis'] = payment_kpis
    
    # ============================================================
    # KPIs DE CALIDAD (REVIEWS)
    # ============================================================
    if 'review_score' in master_df.columns:
        logger.info("")
        logger.info("⭐ KPIs DE CALIDAD (REVIEWS)")
        logger.info("-" * 80)
        
        late_reviews = master_df[master_df['is_late'] == 1]['review_score'].mean()
        on_time_reviews = master_df[master_df['is_late'] == 0]['review_score'].mean()
        overall_reviews = master_df['review_score'].mean()
        
        kpis['avg_review_late'] = float(late_reviews) if not pd.isna(late_reviews) else 0
        kpis['avg_review_on_time'] = float(on_time_reviews) if not pd.isna(on_time_reviews) else 0
        kpis['avg_review_overall'] = float(overall_reviews) if not pd.isna(overall_reviews) else 0
        kpis['review_impact'] = kpis['avg_review_on_time'] - kpis['avg_review_late']
        
        logger.info(f"   Score promedio general: {overall_reviews:.2f} ⭐")
        logger.info(f"   Score promedio (entregas a tiempo): {on_time_reviews:.2f} ⭐")
        logger.info(f"   Score promedio (entregas tardías): {late_reviews:.2f} ⭐")
        logger.info(f"   Impacto del retraso en reviews: {kpis['review_impact']:.2f} ⭐")
    
    # ============================================================
    # KPIs POR RANGO DE PRECIO
    # ============================================================
    if 'price' in master_df.columns:
        logger.info("")
        logger.info("💰 KPIs POR RANGO DE PRECIO")
        logger.info("-" * 80)
        
        price_ranges = pd.cut(master_df['price'], 
                             bins=[0, 50, 100, 200, 500, float('inf')],
                             labels=['0-50', '50-100', '100-200', '200-500', '500+'])
        
        master_df['price_range'] = price_ranges
        
        price_kpis = master_df.groupby('price_range').agg({
            'is_late': ['count', 'sum', 'mean'],
            'Delayed_time': 'mean'
        }).round(2)
        
        price_kpis.columns = ['total_orders', 'late_orders', 'late_rate', 'avg_delay']
        price_kpis['late_rate'] = price_kpis['late_rate'] * 100
        
        logger.info(f"\n{price_kpis.to_string()}")
        kpis['price_kpis'] = price_kpis
    
    # ============================================================
    # GUARDAR KPIs
    # ============================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("💾 GUARDANDO KPIs EN OUTPUTS")
    logger.info("=" * 80)
    
    # Timestamp para archivos
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Resumen de KPIs principales
    kpis_summary = pd.DataFrame([
        {'KPI': 'Total de Órdenes', 'Valor': f"{kpis['total_orders']:,}"},
        {'KPI': 'Tasa de Entregas Tardías', 'Valor': f"{kpis['late_rate']:.2f}%"},
        {'KPI': 'Tasa de Entregas a Tiempo', 'Valor': f"{kpis['on_time_rate']:.2f}%"},
        {'KPI': 'Días Promedio de Retraso', 'Valor': f"{kpis.get('avg_delay_days', 0):.2f}"},
        {'KPI': 'Mediana de Retraso', 'Valor': f"{kpis.get('median_delay_days', 0):.2f}"},
        {'KPI': 'Review Promedio (A Tiempo)', 'Valor': f"{kpis.get('avg_review_on_time', 0):.2f} ⭐"},
        {'KPI': 'Review Promedio (Tardías)', 'Valor': f"{kpis.get('avg_review_late', 0):.2f} ⭐"},
        {'KPI': 'Impacto Retraso en Reviews', 'Valor': f"{kpis.get('review_impact', 0):.2f} ⭐"},
        {'KPI': 'MAE del Modelo', 'Valor': f"{kpis.get('model_mae', 0):.2f} días"},
        {'KPI': 'Precisión del Modelo', 'Valor': f"{kpis.get('model_accuracy', 0):.2f}%"},
    ])
    
    summary_path = os.path.join(OUTPUT_DIR, f'business_kpis_summary_{timestamp}.csv')
    kpis_summary.to_csv(summary_path, index=False)
    logger.info(f"   ✅ Resumen de KPIs: {summary_path}")
    
    # 2. KPIs detallados por dimensión
    if 'state_kpis' in kpis:
        state_path = os.path.join(OUTPUT_DIR, f'kpis_by_state_{timestamp}.csv')
        kpis['state_kpis'].to_csv(state_path)
        logger.info(f"   ✅ KPIs por estado: {state_path}")
    
    if 'payment_kpis' in kpis:
        payment_path = os.path.join(OUTPUT_DIR, f'kpis_by_payment_{timestamp}.csv')
        kpis['payment_kpis'].to_csv(payment_path)
        logger.info(f"   ✅ KPIs por método de pago: {payment_path}")
    
    if 'price_kpis' in kpis:
        price_path = os.path.join(OUTPUT_DIR, f'kpis_by_price_{timestamp}.csv')
        kpis['price_kpis'].to_csv(price_path)
        logger.info(f"   ✅ KPIs por rango de precio: {price_path}")
    
    logger.info("=" * 80)
    
    return kpis