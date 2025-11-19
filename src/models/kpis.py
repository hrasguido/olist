# src/models/kpis.py
"""Módulo para cálculo de KPIs de negocio."""

import pandas as pd
import os


def calculate_business_kpis(master_df, initial_master_df, output_dir='outputs/'):
    """Calcula y muestra KPIs de negocio derivados del análisis.
    
    Args:
        master_df: DataFrame con features y targets
        initial_master_df: DataFrame inicial con datos completos
        output_dir: Directorio para guardar resultados
    
    Returns:
        dict con todos los KPIs calculados
    """
    print("\n" + "="*70)
    print("  KPIs DE NEGOCIO")
    print("="*70)
    
    kpis = {}
    
    # KPIs Operacionales Generales
    print("\n📊 KPIs OPERACIONALES GENERALES")
    print("-" * 70)
    
    total_orders = len(master_df)
    late_orders = master_df['is_late_delivery'].sum()
    on_time_orders = total_orders - late_orders
    late_rate = (late_orders / total_orders) * 100
    on_time_rate = 100 - late_rate
    
    kpis['total_orders'] = total_orders
    kpis['late_orders'] = late_orders
    kpis['on_time_orders'] = on_time_orders
    kpis['late_rate'] = late_rate
    kpis['on_time_rate'] = on_time_rate
    
    print(f"Total de órdenes analizadas: {total_orders:,}")
    print(f"Entregas tardías: {late_orders:,} ({late_rate:.2f}%)")
    print(f"Entregas a tiempo: {on_time_orders:,} ({on_time_rate:.2f}%)")
    
    # KPIs de Retraso
    print("\n⏱️  KPIs DE RETRASO")
    print("-" * 70)
    
    late_deliveries = master_df[master_df['is_late_delivery'] == 1]
    if len(late_deliveries) > 0:
        avg_delay = late_deliveries['delay_days'].mean()
        median_delay = late_deliveries['delay_days'].median()
        max_delay = late_deliveries['delay_days'].max()
        min_delay = late_deliveries['delay_days'].min()
        std_delay = late_deliveries['delay_days'].std()
        
        kpis['avg_delay_days'] = avg_delay
        kpis['median_delay_days'] = median_delay
        kpis['max_delay_days'] = max_delay
        kpis['min_delay_days'] = min_delay
        kpis['std_delay_days'] = std_delay
        
        print(f"Días promedio de retraso: {avg_delay:.2f} días")
        print(f"Mediana de retraso: {median_delay:.2f} días")
        print(f"Retraso máximo: {max_delay:.2f} días")
        print(f"Retraso mínimo: {min_delay:.2f} días")
        print(f"Desviación estándar: {std_delay:.2f} días")
    
    # KPIs por Categoría de Producto
    print("\n🏷️  KPIs POR CATEGORÍA DE PRODUCTO (Top 10)")
    print("-" * 70)
    
    if 'product_category_name_english' in initial_master_df.columns:
        category_kpis = initial_master_df.merge(
            master_df[['is_late_delivery', 'delay_days']], 
            left_index=True, 
            right_index=True, 
            how='inner'
        ).groupby('product_category_name_english').agg({
            'is_late_delivery': ['count', 'sum', 'mean'],
            'delay_days': 'mean'
        }).round(2)
        
        category_kpis.columns = ['total_orders', 'late_orders', 'late_rate', 'avg_delay']
        category_kpis = category_kpis.sort_values('late_rate', ascending=False).head(10)
        category_kpis['late_rate'] = category_kpis['late_rate'] * 100
        
        print(category_kpis.to_string())
        kpis['category_kpis'] = category_kpis
    
    # KPIs por Estado (Seller)
    print("\n🗺️  KPIs POR ESTADO DEL VENDEDOR (Top 10)")
    print("-" * 70)
    
    if 'seller_state' in initial_master_df.columns:
        state_kpis = initial_master_df.merge(
            master_df[['is_late_delivery', 'delay_days']], 
            left_index=True, 
            right_index=True, 
            how='inner'
        ).groupby('seller_state').agg({
            'is_late_delivery': ['count', 'sum', 'mean'],
            'delay_days': 'mean'
        }).round(2)
        
        state_kpis.columns = ['total_orders', 'late_orders', 'late_rate', 'avg_delay']
        state_kpis = state_kpis.sort_values('late_rate', ascending=False).head(10)
        state_kpis['late_rate'] = state_kpis['late_rate'] * 100
        
        print(state_kpis.to_string())
        kpis['state_kpis'] = state_kpis
    
    # KPIs por Método de Pago
    print("\n💳 KPIs POR MÉTODO DE PAGO")
    print("-" * 70)
    
    if 'payment_type' in initial_master_df.columns:
        payment_kpis = initial_master_df.merge(
            master_df[['is_late_delivery', 'delay_days']], 
            left_index=True, 
            right_index=True, 
            how='inner'
        ).groupby('payment_type').agg({
            'is_late_delivery': ['count', 'sum', 'mean'],
            'delay_days': 'mean'
        }).round(2)
        
        payment_kpis.columns = ['total_orders', 'late_orders', 'late_rate', 'avg_delay']
        payment_kpis = payment_kpis.sort_values('late_rate', ascending=False)
        payment_kpis['late_rate'] = payment_kpis['late_rate'] * 100
        
        print(payment_kpis.to_string())
        kpis['payment_kpis'] = payment_kpis
    
    # KPIs Temporales
    print("\n📅 KPIs TEMPORALES")
    print("-" * 70)
    
    # Por día de la semana
    if 'purchase_day_of_week' in master_df.columns:
        day_names = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves', 
                     4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
        
        temporal_kpis = master_df.groupby('purchase_day_of_week').agg({
            'is_late_delivery': ['count', 'sum', 'mean'],
            'delay_days': 'mean'
        }).round(2)
        
        temporal_kpis.columns = ['total_orders', 'late_orders', 'late_rate', 'avg_delay']
        temporal_kpis.index = temporal_kpis.index.map(day_names)
        temporal_kpis['late_rate'] = temporal_kpis['late_rate'] * 100
        
        print("\nPor día de la semana:")
        print(temporal_kpis.to_string())
        kpis['temporal_kpis'] = temporal_kpis
    
    # Por mes
    if 'purchase_month' in master_df.columns:
        month_names = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                      5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                      9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
        
        month_kpis = master_df.groupby('purchase_month').agg({
            'is_late_delivery': ['count', 'sum', 'mean'],
            'delay_days': 'mean'
        }).round(2)
        
        month_kpis.columns = ['total_orders', 'late_orders', 'late_rate', 'avg_delay']
        month_kpis.index = month_kpis.index.map(month_names)
        month_kpis['late_rate'] = month_kpis['late_rate'] * 100
        
        print("\nPor mes:")
        print(month_kpis.to_string())
        kpis['month_kpis'] = month_kpis
    
    # KPIs de Calidad (Reviews)
    print("\n⭐ KPIs DE CALIDAD (REVIEWS)")
    print("-" * 70)
    
    if 'review_score' in initial_master_df.columns:
        review_comparison = initial_master_df.merge(
            master_df[['is_late_delivery']], 
            left_index=True, 
            right_index=True, 
            how='inner'
        )
        
        late_reviews = review_comparison[review_comparison['is_late_delivery'] == 1]['review_score'].mean()
        on_time_reviews = review_comparison[review_comparison['is_late_delivery'] == 0]['review_score'].mean()
        overall_reviews = review_comparison['review_score'].mean()
        
        kpis['avg_review_late'] = late_reviews
        kpis['avg_review_on_time'] = on_time_reviews
        kpis['avg_review_overall'] = overall_reviews
        kpis['review_impact'] = on_time_reviews - late_reviews
        
        print(f"Score promedio general: {overall_reviews:.2f} ⭐")
        print(f"Score promedio (entregas a tiempo): {on_time_reviews:.2f} ⭐")
        print(f"Score promedio (entregas tardías): {late_reviews:.2f} ⭐")
        print(f"Impacto del retraso en reviews: -{kpis['review_impact']:.2f} ⭐")
    
    # KPIs por Rango de Precio
    print("\n💰 KPIs POR RANGO DE PRECIO")
    print("-" * 70)
    
    if 'price' in initial_master_df.columns:
        price_ranges = pd.cut(initial_master_df['price'], 
                             bins=[0, 50, 100, 200, 500, float('inf')],
                             labels=['0-50', '50-100', '100-200', '200-500', '500+'])
        
        price_df = initial_master_df.copy()
        price_df['price_range'] = price_ranges
        
        price_kpis = price_df.merge(
            master_df[['is_late_delivery', 'delay_days']], 
            left_index=True, 
            right_index=True, 
            how='inner'
        ).groupby('price_range').agg({
            'is_late_delivery': ['count', 'sum', 'mean'],
            'delay_days': 'mean'
        }).round(2)
        
        price_kpis.columns = ['total_orders', 'late_orders', 'late_rate', 'avg_delay']
        price_kpis['late_rate'] = price_kpis['late_rate'] * 100
        
        print(price_kpis.to_string())
        kpis['price_kpis'] = price_kpis
    
    # Guardar KPIs en CSV
    kpis_summary = pd.DataFrame([
        {'KPI': 'Total de Órdenes', 'Valor': f"{kpis['total_orders']:,}"},
        {'KPI': 'Tasa de Entregas Tardías', 'Valor': f"{kpis['late_rate']:.2f}%"},
        {'KPI': 'Tasa de Entregas a Tiempo', 'Valor': f"{kpis['on_time_rate']:.2f}%"},
        {'KPI': 'Días Promedio de Retraso', 'Valor': f"{kpis.get('avg_delay_days', 0):.2f}"},
        {'KPI': 'Review Promedio (A Tiempo)', 'Valor': f"{kpis.get('avg_review_on_time', 0):.2f} ⭐"},
        {'KPI': 'Review Promedio (Tardías)', 'Valor': f"{kpis.get('avg_review_late', 0):.2f} ⭐"},
        {'KPI': 'Impacto Retraso en Reviews', 'Valor': f"{kpis.get('review_impact', 0):.2f} ⭐"},
    ])
    
    kpis_path = os.path.join(output_dir, 'business_kpis_summary.csv')
    kpis_summary.to_csv(kpis_path, index=False)
    print(f"\n✅ KPIs guardados en: {kpis_path}")
    
    # Guardar KPIs detallados
    if 'category_kpis' in kpis:
        kpis['category_kpis'].to_csv(os.path.join(output_dir, 'kpis_by_category.csv'))
    if 'state_kpis' in kpis:
        kpis['state_kpis'].to_csv(os.path.join(output_dir, 'kpis_by_state.csv'))
    if 'payment_kpis' in kpis:
        kpis['payment_kpis'].to_csv(os.path.join(output_dir, 'kpis_by_payment.csv'))
    if 'temporal_kpis' in kpis:
        kpis['temporal_kpis'].to_csv(os.path.join(output_dir, 'kpis_by_day.csv'))
    if 'month_kpis' in kpis:
        kpis['month_kpis'].to_csv(os.path.join(output_dir, 'kpis_by_month.csv'))
    if 'price_kpis' in kpis:
        kpis['price_kpis'].to_csv(os.path.join(output_dir, 'kpis_by_price.csv'))
    
    print("\n" + "="*70)
    
    return kpis