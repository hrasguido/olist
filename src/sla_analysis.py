"""
Módulo para análisis de SLA (Service Level Agreement).
Compara tiempos de entrega reales vs. SLA prometido.
Incluye análisis de predicciones del modelo ML.
"""

from prefect import task, get_run_logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
sns.set_style("whitegrid")

# Configurar carpeta de outputs (ruta relativa)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


@task(log_prints=True)
def calculate_sla_metrics(master_df: pd.DataFrame) -> Dict:
    """
    Calcula métricas de SLA basadas en estimated_delivery_date.
    Incluye análisis de predicciones del modelo si están disponibles.
    
    Args:
        master_df: DataFrame con datos de órdenes
    
    Returns:
        Dict con métricas de SLA, predicción y DataFrame enriquecido
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("📋 CALCULANDO MÉTRICAS DE SLA")
    logger.info("=" * 80)
    
    # ============================================================
    # VERIFICAR Y MAPEAR COLUMNAS
    # ============================================================
    # Mapeo de nombres de columnas (pueden variar)
    column_mapping = {
        'order_approved_at': ['order_approved_at', 'order_purchase_timestamp'],
        'estimated_delivery_date': ['order_estimated_delivery_date', 'estimated_delivery_date'],
        'order_delivered_customer_date': ['order_delivered_customer_date']
    }
    
    # Encontrar columnas reales
    actual_columns = {}
    for key, possible_names in column_mapping.items():
        found = False
        for name in possible_names:
            if name in master_df.columns:
                actual_columns[key] = name
                found = True
                break
        if not found:
            logger.warning(f"⚠️  No se encontró columna para '{key}'. Opciones buscadas: {possible_names}")
            logger.warning(f"⚠️  Columnas disponibles en master_df: {list(master_df.columns)[:20]}...")
            return {}
    
    logger.info(f"✅ Columnas mapeadas:")
    for key, col in actual_columns.items():
        logger.info(f"   • {key} → {col}")
    
    # Verificar si existe columna de predicción
    has_predictions = 'Delayed_time_predicted' in master_df.columns
    if has_predictions:
        logger.info(f"   • ✅ Delayed_time_predicted → Análisis de predicción habilitado")
    else:
        logger.info(f"   • ⚠️  Delayed_time_predicted no encontrada → Análisis de predicción omitido")
    
    # Crear copia para no modificar el original
    df = master_df.copy()
    
    # ============================================================
    # CALCULAR TIEMPOS
    # ============================================================
    logger.info("")
    logger.info("⏱️  Calculando tiempos de entrega...")
    
    # Convertir a datetime usando los nombres reales de columnas
    df['order_approved_at'] = pd.to_datetime(df[actual_columns['order_approved_at']])
    df['estimated_delivery_date'] = pd.to_datetime(df[actual_columns['estimated_delivery_date']])
    df['order_delivered_customer_date'] = pd.to_datetime(df[actual_columns['order_delivered_customer_date']])
    
    # 1. SLA Prometido (max_allowed_delivery_time)
    df['sla_promised_days'] = (
        df['estimated_delivery_date'] - df['order_approved_at']
    ).dt.total_seconds() / (24 * 3600)
    
    # 2. Tiempo Real de Entrega
    df['delivery_time_real_days'] = (
        df['order_delivered_customer_date'] - df['order_approved_at']
    ).dt.total_seconds() / (24 * 3600)
    
    # 3. Diferencia: Real - SLA (positivo = violación de SLA)
    df['sla_deviation_days'] = df['delivery_time_real_days'] - df['sla_promised_days']
    
    # 4. Clasificación de SLA
    df['sla_status'] = 'Unknown'
    df.loc[df['sla_deviation_days'] <= 0, 'sla_status'] = 'On Time'  # A tiempo o antes
    df.loc[df['sla_deviation_days'] > 0, 'sla_status'] = 'Violated'  # Violación de SLA
    
    # ============================================================
    # ANÁLISIS DE PREDICCIÓN (si existe)
    # ============================================================
    if has_predictions:
        logger.info("   🤖 Calculando métricas de predicción del modelo...")
        
        # Fecha de entrega predicha
        df['predicted_delivery_date'] = df['estimated_delivery_date'] + pd.to_timedelta(df['Delayed_time_predicted'], unit='D')
        
        # Tiempo de entrega predicho (desde aprobación)
        df['delivery_time_predicted_days'] = (
            df['predicted_delivery_date'] - df['order_approved_at']
        ).dt.total_seconds() / (24 * 3600)
        
        # Desviación predicha vs SLA
        df['sla_deviation_predicted'] = df['delivery_time_predicted_days'] - df['sla_promised_days']
        
        # Clasificación SLA basada en predicción
        df['sla_status_predicted'] = 'Unknown'
        df.loc[df['sla_deviation_predicted'] <= 0, 'sla_status_predicted'] = 'On Time'
        df.loc[df['sla_deviation_predicted'] > 0, 'sla_status_predicted'] = 'Violated'
        
        # Error de predicción (solo para entregas completadas)
        df['prediction_error_sla'] = df['delivery_time_real_days'] - df['delivery_time_predicted_days']
    
    # Filtrar solo entregas completadas
    completed = df[df['order_delivered_customer_date'].notna()].copy()
    
    logger.info(f"   • Total de órdenes: {len(df):,}")
    logger.info(f"   • Órdenes completadas: {len(completed):,}")
    
    # ============================================================
    # MÉTRICAS DE SLA
    # ============================================================
    logger.info("")
    logger.info("📊 MÉTRICAS DE SLA")
    logger.info("-" * 80)
    
    total_completed = len(completed)
    
    if total_completed == 0:
        logger.warning("⚠️  No hay entregas completadas para analizar")
        return {}
    
    on_time = (completed['sla_status'] == 'On Time').sum()
    violated = (completed['sla_status'] == 'Violated').sum()
    
    on_time_rate = (on_time / total_completed * 100) if total_completed > 0 else 0
    violation_rate = (violated / total_completed * 100) if total_completed > 0 else 0
    
    # Estadísticas de desviación
    avg_deviation = completed['sla_deviation_days'].mean()
    median_deviation = completed['sla_deviation_days'].median()
    
    # Desviación solo para violaciones
    violations_only = completed[completed['sla_status'] == 'Violated']
    if len(violations_only) > 0:
        avg_violation_days = violations_only['sla_deviation_days'].mean()
        max_violation_days = violations_only['sla_deviation_days'].max()
    else:
        avg_violation_days = 0
        max_violation_days = 0
    
    # Estadísticas de SLA prometido
    avg_sla_promised = completed['sla_promised_days'].mean()
    median_sla_promised = completed['sla_promised_days'].median()
    
    # Estadísticas de tiempo real
    avg_delivery_real = completed['delivery_time_real_days'].mean()
    median_delivery_real = completed['delivery_time_real_days'].median()
    
    logger.info(f"   📦 Total de entregas completadas: {total_completed:,}")
    logger.info(f"")
    logger.info(f"   ✅ Entregas a tiempo (SLA cumplido): {on_time:,} ({on_time_rate:.2f}%)")
    logger.info(f"   ❌ Violaciones de SLA: {violated:,} ({violation_rate:.2f}%)")
    logger.info(f"")
    logger.info(f"   ⏱️  SLA PROMETIDO:")
    logger.info(f"      • Promedio: {avg_sla_promised:.2f} días")
    logger.info(f"      • Mediana: {median_sla_promised:.2f} días")
    logger.info(f"")
    logger.info(f"   📅 TIEMPO REAL DE ENTREGA:")
    logger.info(f"      • Promedio: {avg_delivery_real:.2f} días")
    logger.info(f"      • Mediana: {median_delivery_real:.2f} días")
    logger.info(f"")
    logger.info(f"   📊 DESVIACIÓN DEL SLA:")
    logger.info(f"      • Promedio: {avg_deviation:.2f} días")
    logger.info(f"      • Mediana: {median_deviation:.2f} días")
    logger.info(f"      • Promedio (solo violaciones): {avg_violation_days:.2f} días")
    logger.info(f"      • Máxima violación: {max_violation_days:.2f} días")
    
    # ============================================================
    # MÉTRICAS DE PREDICCIÓN (si existen)
    # ============================================================
    if has_predictions:
        completed_pred = completed[completed['Delayed_time_predicted'].notna()].copy()
        
        if len(completed_pred) > 0:
            # Predicciones de SLA
            pred_on_time = (completed_pred['sla_status_predicted'] == 'On Time').sum()
            pred_violated = (completed_pred['sla_status_predicted'] == 'Violated').sum()
            pred_on_time_rate = (pred_on_time / len(completed_pred)) * 100
            pred_violation_rate = (pred_violated / len(completed_pred)) * 100
            
            # Error de predicción
            mae_sla = completed_pred['prediction_error_sla'].abs().mean()
            rmse_sla = np.sqrt((completed_pred['prediction_error_sla'] ** 2).mean())
            
            # Accuracy de clasificación SLA
            correct_predictions = (completed_pred['sla_status'] == completed_pred['sla_status_predicted']).sum()
            sla_accuracy = (correct_predictions / len(completed_pred)) * 100
            
            # Promedio de tiempo predicho
            avg_delivery_predicted = completed_pred['delivery_time_predicted_days'].mean()
            median_delivery_predicted = completed_pred['delivery_time_predicted_days'].median()
            
            logger.info(f"")
            logger.info(f"   🤖 PREDICCIONES DEL MODELO:")
            logger.info(f"      • Órdenes con predicción: {len(completed_pred):,}")
            logger.info(f"      • Predicción 'On Time': {pred_on_time:,} ({pred_on_time_rate:.2f}%)")
            logger.info(f"      • Predicción 'Violated': {pred_violated:,} ({pred_violation_rate:.2f}%)")
            logger.info(f"")
            logger.info(f"   📅 TIEMPO PREDICHO DE ENTREGA:")
            logger.info(f"      • Promedio: {avg_delivery_predicted:.2f} días")
            logger.info(f"      • Mediana: {median_delivery_predicted:.2f} días")
            logger.info(f"")
            logger.info(f"   🎯 PRECISIÓN DE PREDICCIÓN SLA:")
            logger.info(f"      • Accuracy clasificación: {sla_accuracy:.2f}%")
            logger.info(f"      • MAE predicción: {mae_sla:.2f} días")
            logger.info(f"      • RMSE predicción: {rmse_sla:.2f} días")
        else:
            logger.warning(f"   ⚠️  No hay predicciones válidas para analizar")
            completed_pred = None
    else:
        completed_pred = None
    
    # ============================================================
    # PREPARAR RESULTADOS
    # ============================================================
    metrics = {
        'total_completed': int(total_completed),
        'on_time_count': int(on_time),
        'violated_count': int(violated),
        'on_time_rate': float(on_time_rate),
        'violation_rate': float(violation_rate),
        'avg_sla_promised': float(avg_sla_promised),
        'median_sla_promised': float(median_sla_promised),
        'avg_delivery_real': float(avg_delivery_real),
        'median_delivery_real': float(median_delivery_real),
        'avg_deviation': float(avg_deviation),
        'median_deviation': float(median_deviation),
        'avg_violation_days': float(avg_violation_days),
        'max_violation_days': float(max_violation_days),
        'has_predictions': has_predictions
    }
    
    # Agregar métricas de predicción si existen
    if has_predictions and completed_pred is not None and len(completed_pred) > 0:
        metrics.update({
            'predicted_on_time_count': int(pred_on_time),
            'predicted_violated_count': int(pred_violated),
            'predicted_on_time_rate': float(pred_on_time_rate),
            'predicted_violation_rate': float(pred_violation_rate),
            'sla_classification_accuracy': float(sla_accuracy),
            'prediction_mae_sla': float(mae_sla),
            'prediction_rmse_sla': float(rmse_sla),
            'avg_delivery_predicted': float(avg_delivery_predicted),
            'median_delivery_predicted': float(median_delivery_predicted),
            'total_with_predictions': int(len(completed_pred))
        })
    
    logger.info("=" * 80)
    
    return {
        'metrics': metrics,
        'df_with_sla': completed
    }


@task(log_prints=True)
def plot_sla_analysis(
    df_with_sla: pd.DataFrame,
    sla_metrics: Dict,
    output_path: str = None
) -> str:
    """
    Genera gráficos completos de análisis de SLA.
    Incluye gráfico de predicción si está disponible.
    
    Args:
        df_with_sla: DataFrame con métricas de SLA calculadas
        sla_metrics: Dict con métricas agregadas
        output_path: Ruta del archivo de salida
    
    Returns:
        Ruta del archivo guardado
    """
    logger = get_run_logger()
    logger.info("📊 Generando gráficos de análisis de SLA...")
    
    # Definir ruta por defecto con timestamp
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(OUTPUT_DIR, f'sla_analysis_plot_{timestamp}.png')
    
    # Verificar si hay predicciones
    has_predictions = sla_metrics.get('has_predictions', False) and 'delivery_time_predicted_days' in df_with_sla.columns
    
    # Crear figura con subplots (ajustar tamaño si hay predicciones)
    if has_predictions:
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)
    else:
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # ============================================================
    # 1. PIE CHART: Cumplimiento de SLA
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    sizes = [sla_metrics['on_time_count'], sla_metrics['violated_count']]
    labels = [f"A Tiempo\n{sla_metrics['on_time_rate']:.1f}%", 
              f"Retraso\n{sla_metrics['violation_rate']:.1f}%"]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Cumplimiento de SLA (Real)', fontsize=13, fontweight='bold', pad=15)
    
    # ============================================================
    # 2. BAR CHART: Comparación Real vs SLA
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1:])
    
    if has_predictions:
        categories = ['SLA Prometido', 'Tiempo Predicho', 'Tiempo Real']
        means = [sla_metrics['avg_sla_promised'], 
                sla_metrics.get('avg_delivery_predicted', 0), 
                sla_metrics['avg_delivery_real']]
        medians = [sla_metrics['median_sla_promised'], 
                  sla_metrics.get('median_delivery_predicted', 0),
                  sla_metrics['median_delivery_real']]
    else:
        categories = ['SLA Prometido', 'Tiempo Real']
        means = [sla_metrics['avg_sla_promised'], sla_metrics['avg_delivery_real']]
        medians = [sla_metrics['median_sla_promised'], sla_metrics['median_delivery_real']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, means, width, label='Promedio', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, medians, width, label='Mediana', color='coral', alpha=0.8)
    
    ax2.set_ylabel('Días', fontsize=11, fontweight='bold')
    ax2.set_title('Comparación: SLA vs Predicho vs Real', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=15 if has_predictions else 0)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}d', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}d', ha='center', va='bottom', fontsize=9)
    
    # ============================================================
    # 3. HISTOGRAM: Distribución de Desviación del SLA
    # ============================================================
    ax3 = fig.add_subplot(gs[1, :])
    
    deviations = df_with_sla['sla_deviation_days']
    
    # Histograma
    n, bins, patches = ax3.hist(deviations, bins=60, color='skyblue', 
                                 alpha=0.7, edgecolor='black')
    
    # Colorear barras: verde (a tiempo), rojo (violado)
    for i, patch in enumerate(patches):
        if bins[i] <= 0:
            patch.set_facecolor('#2ecc71')
        else:
            patch.set_facecolor('#e74c3c')
    
    # Línea vertical en 0 (límite de SLA)
    ax3.axvline(0, color='black', linestyle='--', linewidth=2, label='Límite SLA')
    ax3.axvline(deviations.mean(), color='blue', linestyle='--', linewidth=2, 
                label=f'Media: {deviations.mean():.2f}d')
    ax3.axvline(deviations.median(), color='orange', linestyle='--', linewidth=2,
                label=f'Mediana: {deviations.median():.2f}d')
    
    ax3.set_xlabel('Desviación del SLA (días)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax3.set_title('Distribución de Desviación del SLA (Negativo = A tiempo, Positivo = Violado)', 
                  fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ============================================================
    # 4. BOX PLOT: SLA Prometido vs Tiempo Real
    # ============================================================
    ax4 = fig.add_subplot(gs[2, 0])
    
    if has_predictions:
        data_to_plot = [df_with_sla['sla_promised_days'], 
                       df_with_sla['delivery_time_predicted_days'].dropna(),
                       df_with_sla['delivery_time_real_days']]
        labels_box = ['SLA Prometido', 'Predicho', 'Real']
        colors_box = ['lightblue', 'lightyellow', 'lightcoral']
    else:
        data_to_plot = [df_with_sla['sla_promised_days'], df_with_sla['delivery_time_real_days']]
        labels_box = ['SLA Prometido', 'Tiempo Real']
        colors_box = ['lightblue', 'lightcoral']
    
    bp = ax4.boxplot(data_to_plot, labels=labels_box,
                     patch_artist=True, showmeans=True)
    
    # Colorear cajas
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('Días', fontsize=11, fontweight='bold')
    ax4.set_title('Distribución: SLA vs Predicho vs Real', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(axis='x', rotation=15)
    
    # ============================================================
    # 5. SCATTER: SLA Prometido vs Tiempo Real
    # ============================================================
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Sample para mejor visualización
    sample_size = min(3000, len(df_with_sla))
    sample_df = df_with_sla.sample(n=sample_size, random_state=42)
    
    # Colorear por estado de SLA
    colors_scatter = sample_df['sla_status'].map({'On Time': '#2ecc71', 'Violated': '#e74c3c'})
    
    ax5.scatter(sample_df['sla_promised_days'], sample_df['delivery_time_real_days'],
                alpha=0.5, s=20, c=colors_scatter, edgecolors='none')
    
    # Línea de igualdad (SLA = Real)
    max_val = max(sample_df['sla_promised_days'].max(), sample_df['delivery_time_real_days'].max())
    min_val = min(sample_df['sla_promised_days'].min(), sample_df['delivery_time_real_days'].min())
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='SLA = Real')
    
    ax5.set_xlabel('SLA Prometido (días)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Tiempo Real (días)', fontsize=11, fontweight='bold')
    ax5.set_title('SLA Prometido vs Tiempo Real', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # ============================================================
    # 6. BAR CHART: Violaciones por Rango de Días
    # ============================================================
    ax6 = fig.add_subplot(gs[2, 2])
    
    violations = df_with_sla[df_with_sla['sla_status'] == 'Violated']
    
    if len(violations) > 0:
        # Crear rangos de violación
        bins_violation = [0, 1, 3, 7, 14, 30, float('inf')]
        labels_violation = ['0-1d', '1-3d', '3-7d', '7-14d', '14-30d', '>30d']
        
        violations['violation_range'] = pd.cut(violations['sla_deviation_days'], 
                                               bins=bins_violation, 
                                               labels=labels_violation)
        
        violation_counts = violations['violation_range'].value_counts().sort_index()
        
        bars = ax6.bar(range(len(violation_counts)), violation_counts.values,
                       color=sns.color_palette("Reds_r", len(violation_counts)), alpha=0.8)
        
        ax6.set_xticks(range(len(violation_counts)))
        ax6.set_xticklabels(violation_counts.index, rotation=45)
        ax6.set_ylabel('Cantidad', fontsize=11, fontweight='bold')
        ax6.set_title('Violaciones de SLA por Rango', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Agregar valores
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    # ============================================================
    # 7. SCATTER: Predicción vs Real (si existe)
    # ============================================================
    if has_predictions:
        ax7 = fig.add_subplot(gs[3, 0])
        
        pred_data = df_with_sla[df_with_sla['delivery_time_predicted_days'].notna()].copy()
        
        if len(pred_data) > 0:
            # Sample para visualización
            sample_size_pred = min(3000, len(pred_data))
            sample_pred = pred_data.sample(n=sample_size_pred, random_state=42)
            
            # Scatter plot coloreado por error absoluto
            scatter = ax7.scatter(
                sample_pred['delivery_time_predicted_days'],
                sample_pred['delivery_time_real_days'],
                c=sample_pred['prediction_error_sla'].abs(),
                cmap='RdYlGn_r',
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Línea de referencia (predicción perfecta)
            min_val_pred = min(sample_pred['delivery_time_predicted_days'].min(),
                              sample_pred['delivery_time_real_days'].min())
            max_val_pred = max(sample_pred['delivery_time_predicted_days'].max(),
                              sample_pred['delivery_time_real_days'].max())
            ax7.plot([min_val_pred, max_val_pred], [min_val_pred, max_val_pred], 
                    'r--', linewidth=2, label='Predicción Perfecta', alpha=0.7)
            
            ax7.set_xlabel('Tiempo Predicho (días)', fontsize=11, fontweight='bold')
            ax7.set_ylabel('Tiempo Real (días)', fontsize=11, fontweight='bold')
            ax7.set_title('Predicción vs Realidad', fontsize=13, fontweight='bold', pad=15)
            ax7.grid(True, alpha=0.3, linestyle='--')
            ax7.legend(fontsize=9)
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax7)
            cbar.set_label('Error Absoluto (días)', fontsize=9)
        
        # ============================================================
        # 8. PIE CHART: Cumplimiento SLA Predicho
        # ============================================================
        ax8 = fig.add_subplot(gs[3, 1])
        
        if 'predicted_on_time_count' in sla_metrics and 'predicted_violated_count' in sla_metrics:
            sizes_pred = [sla_metrics['predicted_on_time_count'], sla_metrics['predicted_violated_count']]
            labels_pred = [f"Pred. A Tiempo\n{sla_metrics['predicted_on_time_rate']:.1f}%", 
                          f"Pred. Retraso\n{sla_metrics['predicted_violation_rate']:.1f}%"]
            colors_pred = ['#3498db', '#e67e22']
            explode_pred = (0.05, 0.05)
            
            ax8.pie(sizes_pred, explode=explode_pred, labels=labels_pred, colors=colors_pred,
                    autopct='%1.1f%%', shadow=True, startangle=90,
                    textprops={'fontsize': 11, 'fontweight': 'bold'})
            ax8.set_title('Cumplimiento SLA (Predicho)', fontsize=13, fontweight='bold', pad=15)
        
        # ============================================================
        # 9. BAR CHART: Métricas de Predicción
        # ============================================================
        ax9 = fig.add_subplot(gs[3, 2])
        
        if 'sla_classification_accuracy' in sla_metrics:
            metrics_pred = ['Accuracy\nSLA', 'MAE\n(días)', 'RMSE\n(días)']
            values_pred = [
                sla_metrics['sla_classification_accuracy'],
                sla_metrics['prediction_mae_sla'],
                sla_metrics['prediction_rmse_sla']
            ]
            colors_metrics = ['#2ecc71', '#3498db', '#e74c3c']
            
            bars_pred = ax9.bar(metrics_pred, values_pred, color=colors_metrics, alpha=0.8)
            
            ax9.set_ylabel('Valor', fontsize=11, fontweight='bold')
            ax9.set_title('Métricas de Predicción', fontsize=13, fontweight='bold')
            ax9.grid(True, alpha=0.3, axis='y')
            
            # Agregar valores
            for bar in bars_pred:
                height = bar.get_height()
                ax9.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Título general
    title_text = 'Análisis Completo de SLA (Service Level Agreement)'
    if has_predictions:
        title_text += ' + Predicciones del Modelo ML'
    
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.995)
    
    # Guardar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Gráfico guardado en: {output_path}")
    
    return output_path


@task(log_prints=True)
def save_sla_report(
    df_with_sla: pd.DataFrame,
    sla_metrics: Dict,
    output_path: str = None
) -> str:
    """
    Guarda reporte detallado de SLA en CSV.
    Incluye métricas de predicción si están disponibles.
    
    Args:
        df_with_sla: DataFrame con métricas de SLA
        sla_metrics: Dict con métricas agregadas
        output_path: Ruta del archivo de salida
    
    Returns:
        Ruta del archivo guardado
    """
    logger = get_run_logger()
    logger.info("💾 Guardando reporte de SLA...")
    
    # Definir ruta por defecto con timestamp
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(OUTPUT_DIR, f'sla_report_{timestamp}.csv')
    
    # Crear reporte resumido
    report_data = [
        {'Métrica': 'Total de Entregas Completadas', 'Valor': f"{sla_metrics['total_completed']:,}"},
        {'Métrica': 'Entregas a Tiempo', 'Valor': f"{sla_metrics['on_time_count']:,}"},
        {'Métrica': 'Violaciones de SLA', 'Valor': f"{sla_metrics['violated_count']:,}"},
        {'Métrica': 'Tasa de Cumplimiento', 'Valor': f"{sla_metrics['on_time_rate']:.2f}%"},
        {'Métrica': 'Tasa de Violación', 'Valor': f"{sla_metrics['violation_rate']:.2f}%"},
        {'Métrica': '', 'Valor': ''},
        {'Métrica': 'SLA Prometido (Promedio)', 'Valor': f"{sla_metrics['avg_sla_promised']:.2f} días"},
        {'Métrica': 'SLA Prometido (Mediana)', 'Valor': f"{sla_metrics['median_sla_promised']:.2f} días"},
        {'Métrica': 'Tiempo Real (Promedio)', 'Valor': f"{sla_metrics['avg_delivery_real']:.2f} días"},
        {'Métrica': 'Tiempo Real (Mediana)', 'Valor': f"{sla_metrics['median_delivery_real']:.2f} días"},
        {'Métrica': '', 'Valor': ''},
        {'Métrica': 'Desviación Promedio', 'Valor': f"{sla_metrics['avg_deviation']:.2f} días"},
        {'Métrica': 'Desviación Mediana', 'Valor': f"{sla_metrics['median_deviation']:.2f} días"},
        {'Métrica': 'Desviación Promedio (solo violaciones)', 'Valor': f"{sla_metrics['avg_violation_days']:.2f} días"},
        {'Métrica': 'Máxima Violación', 'Valor': f"{sla_metrics['max_violation_days']:.2f} días"},
    ]
    
    # Agregar métricas de predicción si existen
    if sla_metrics.get('has_predictions', False) and 'sla_classification_accuracy' in sla_metrics:
        report_data.extend([
            {'Métrica': '', 'Valor': ''},
            {'Métrica': '=== PREDICCIONES DEL MODELO ===', 'Valor': ''},
            {'Métrica': 'Órdenes con Predicción', 'Valor': f"{sla_metrics['total_with_predictions']:,}"},
            {'Métrica': 'Tiempo Predicho (Promedio)', 'Valor': f"{sla_metrics['avg_delivery_predicted']:.2f} días"},
            {'Métrica': 'Tiempo Predicho (Mediana)', 'Valor': f"{sla_metrics['median_delivery_predicted']:.2f} días"},
            {'Métrica': '', 'Valor': ''},
            {'Métrica': 'Predicción On Time', 'Valor': f"{sla_metrics['predicted_on_time_count']:,}"},
            {'Métrica': 'Predicción Violated', 'Valor': f"{sla_metrics['predicted_violated_count']:,}"},
            {'Métrica': 'Tasa Predicción On Time', 'Valor': f"{sla_metrics['predicted_on_time_rate']:.2f}%"},
            {'Métrica': 'Tasa Predicción Violated', 'Valor': f"{sla_metrics['predicted_violation_rate']:.2f}%"},
            {'Métrica': '', 'Valor': ''},
            {'Métrica': 'Accuracy Clasificación SLA', 'Valor': f"{sla_metrics['sla_classification_accuracy']:.2f}%"},
            {'Métrica': 'MAE Predicción', 'Valor': f"{sla_metrics['prediction_mae_sla']:.2f} días"},
            {'Métrica': 'RMSE Predicción', 'Valor': f"{sla_metrics['prediction_rmse_sla']:.2f} días"},
        ])
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_path, index=False)
    
    logger.info(f"   ✅ Reporte guardado en: {output_path}")
    
    return output_path


@task(log_prints=True)
def analyze_sla_complete(master_df: pd.DataFrame) -> Dict:
    """
    Análisis completo de SLA: cálculo, gráficos y reportes.
    Incluye análisis de predicciones del modelo si están disponibles.
    
    Args:
        master_df: DataFrame con datos de órdenes (con o sin predicciones)
    
    Returns:
        Dict con resultados del análisis
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("🎯 ANÁLISIS COMPLETO DE SLA")
    logger.info("=" * 80)
    
    # 1. Calcular métricas de SLA
    sla_result = calculate_sla_metrics(master_df)
    
    if not sla_result:
        logger.warning("⚠️  No se pudo realizar el análisis de SLA")
        return {}
    
    # 2. Generar gráficos
    plot_path = plot_sla_analysis(
        sla_result['df_with_sla'],
        sla_result['metrics']
    )
    
    # 3. Guardar reporte
    report_path = save_sla_report(
        sla_result['df_with_sla'],
        sla_result['metrics']
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ ANÁLISIS DE SLA COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"   📊 Gráfico: {plot_path}")
    logger.info(f"   📋 Reporte: {report_path}")
    if sla_result['metrics'].get('has_predictions', False):
        logger.info(f"   🤖 Análisis de predicción: INCLUIDO")
    logger.info("=" * 80)
    
    return {
        'metrics': sla_result['metrics'],
        'plot_path': plot_path,
        'report_path': report_path,
        'df_with_sla': sla_result['df_with_sla']
    }