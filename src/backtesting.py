# src/backtesting.py
"""
Módulo de Backtesting usando dataset de últimos 3 meses.
Valida el modelo con datos fuera del entrenamiento.
"""

from prefect import task, get_run_logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Configurar carpetas
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
BACKTEST_DIR = os.path.join(OUTPUT_DIR, 'backtesting')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(BACKTEST_DIR, exist_ok=True)


@task(log_prints=True)
def load_last_3_months_data() -> pd.DataFrame:
    """
    Carga el dataset de los últimos 3 meses para backtesting.
    """
    logger = get_run_logger()
    logger.info("📥 Cargando dataset de últimos 3 meses...")
    
    csv_path = os.path.join(DATA_DIR, 'olist_orders_last_3_months.csv')
    
    if not os.path.exists(csv_path):
        logger.error(f"   ❌ No se encontró el archivo: {csv_path}")
        raise FileNotFoundError(f"Dataset no encontrado: {csv_path}")
    
    # Cargar CSV
    df = pd.read_csv(csv_path)
    
    logger.info(f"   ✅ Dataset cargado: {len(df):,} registros")
    logger.info(f"   📊 Columnas: {list(df.columns)}")
    
    # Convertir fechas
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calcular Delayed_time si no existe
    if 'Delayed_time' not in df.columns:
        if 'order_delivered_customer_date' in df.columns and 'order_estimated_delivery_date' in df.columns:
            logger.info("   🎯 Calculando Delayed_time...")
            df['Delayed_time'] = (
                df['order_delivered_customer_date'] - 
                df['order_estimated_delivery_date']
            ).dt.days
            logger.info(f"      • Rango: [{df['Delayed_time'].min():.0f}, {df['Delayed_time'].max():.0f}] días")
    
    return df


@task(log_prints=True)
def prepare_backtest_features(
    backtest_df: pd.DataFrame,
    master_df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Prepara las features del dataset de backtesting para que coincidan con el modelo.
    """
    logger = get_run_logger()
    logger.info("🔧 Preparando features para backtesting...")
    
    # Crear copia
    df = backtest_df.copy()
    
    # Asegurar que todas las features existen
    missing_features = []
    for col in feature_cols:
        if col not in df.columns:
            missing_features.append(col)
            df[col] = 0  # Rellenar con 0 si no existe
    
    if missing_features:
        logger.warning(f"   ⚠️  Features faltantes (rellenadas con 0): {len(missing_features)}")
        logger.warning(f"      Primeras 5: {missing_features[:5]}")
    
    # Seleccionar solo las features necesarias
    X_backtest = df[feature_cols].fillna(0)
    
    logger.info(f"   ✅ Features preparadas: {len(feature_cols)} columnas")
    logger.info(f"   📊 Registros: {len(X_backtest):,}")
    
    return X_backtest


@task(log_prints=True)
def perform_backtest_validation(
    model,
    backtest_df: pd.DataFrame,
    master_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'Delayed_time'
) -> Dict:
    """
    Realiza validación del modelo con el dataset de últimos 3 meses.
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("🔄 BACKTESTING CON DATASET DE ÚLTIMOS 3 MESES")
    logger.info("=" * 80)
    
    # Preparar features
    X_backtest = prepare_backtest_features(backtest_df, master_df, feature_cols)
    
    # Verificar que existe el target
    if target_col not in backtest_df.columns:
        logger.error(f"   ❌ Target '{target_col}' no encontrado en dataset de backtest")
        return None
    
    y_backtest = backtest_df[target_col].fillna(0)
    
    # Filtrar outliers (mismo rango que entrenamiento)
    valid_mask = (y_backtest >= -3) & (y_backtest <= 12)
    X_backtest = X_backtest[valid_mask]
    y_backtest = y_backtest[valid_mask]
    backtest_df_filtered = backtest_df[valid_mask].copy()
    
    logger.info(f"   📊 Registros válidos: {len(X_backtest):,}")
    
    # Hacer predicciones
    logger.info("   🔮 Generando predicciones...")
    y_pred = model.predict(X_backtest)
    
    # Calcular métricas
    mae = mean_absolute_error(y_backtest, y_pred)
    rmse = np.sqrt(mean_squared_error(y_backtest, y_pred))
    r2 = r2_score(y_backtest, y_pred)
    
    # Calcular métricas adicionales
    mape = np.mean(np.abs((y_backtest - y_pred) / (y_backtest + 1e-10))) * 100
    median_ae = np.median(np.abs(y_backtest - y_pred))
    
    logger.info("")
    logger.info("   📊 MÉTRICAS DE BACKTESTING:")
    logger.info(f"      • MAE:        {mae:.2f} días")
    logger.info(f"      • RMSE:       {rmse:.2f} días")
    logger.info(f"      • R²:         {r2:.4f}")
    logger.info(f"      • MAPE:       {mape:.2f}%")
    logger.info(f"      • Median AE:  {median_ae:.2f} días")
    
    # Agregar predicciones al dataframe
    backtest_df_filtered['Delayed_time_predicted'] = y_pred
    backtest_df_filtered['prediction_error'] = y_backtest - y_pred
    backtest_df_filtered['prediction_error_abs'] = np.abs(y_backtest - y_pred)
    
    # Análisis de errores por rango
    logger.info("")
    logger.info("   📈 DISTRIBUCIÓN DE ERRORES:")
    error_ranges = [
        ('Excelente (< 1 día)', lambda x: x < 1),
        ('Bueno (1-2 días)', lambda x: (x >= 1) & (x < 2)),
        ('Aceptable (2-3 días)', lambda x: (x >= 2) & (x < 3)),
        ('Malo (> 3 días)', lambda x: x >= 3)
    ]
    
    for label, condition in error_ranges:
        count = condition(backtest_df_filtered['prediction_error_abs']).sum()
        pct = count / len(backtest_df_filtered) * 100
        logger.info(f"      • {label}: {count:,} ({pct:.1f}%)")
    
    logger.info("=" * 80)
    
    return {
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'median_ae': median_ae
        },
        'predictions_df': backtest_df_filtered,
        'n_samples': len(X_backtest)
    }


@task(log_prints=True)
def plot_backtest_comparison(
    backtest_results: Dict,
    train_metrics: Dict,
    output_path: str = None
) -> str:
    """
    Genera gráfico comparando métricas de entrenamiento vs backtesting.
    """
    logger = get_run_logger()
    logger.info("📊 Generando gráfico de comparación Train vs Backtest...")
    
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(BACKTEST_DIR, f'train_vs_backtest_{timestamp}.png')
    
    # Extraer métricas
    train_mae = train_metrics['test']['mae']
    train_rmse = train_metrics['test']['rmse']
    train_r2 = train_metrics['test']['r2']
    
    backtest_mae = backtest_results['metrics']['mae']
    backtest_rmse = backtest_results['metrics']['rmse']
    backtest_r2 = backtest_results['metrics']['r2']
    
    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Comparación: Conjunto de Test vs Backtesting (Últimos 3 Meses)', 
                 fontsize=16, fontweight='bold')
    
    datasets = ['Test Set\n(Entrenamiento)', 'Backtest\n(Últimos 3 Meses)']
    
    # 1. MAE
    ax1 = axes[0]
    mae_values = [train_mae, backtest_mae]
    colors = ['steelblue', 'coral']
    bars = ax1.bar(datasets, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('MAE (días)', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae_values[i]:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Calcular diferencia
    mae_diff = ((backtest_mae - train_mae) / train_mae) * 100
    ax1.text(0.5, 0.95, f'Diferencia: {mae_diff:+.1f}%', 
             transform=ax1.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    # 2. RMSE
    ax2 = axes[1]
    rmse_values = [train_rmse, backtest_rmse]
    bars = ax2.bar(datasets, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('RMSE (días)', fontsize=12, fontweight='bold')
    ax2.set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse_values[i]:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    rmse_diff = ((backtest_rmse - train_rmse) / train_rmse) * 100
    ax2.text(0.5, 0.95, f'Diferencia: {rmse_diff:+.1f}%', 
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    # 3. R²
    ax3 = axes[2]
    r2_values = [train_r2, backtest_r2]
    bars = ax3.bar(datasets, r2_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('R²', fontsize=12, fontweight='bold')
    ax3.set_title('Coeficiente de Determinación', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, max(r2_values) * 1.2])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2_values[i]:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    r2_diff = ((backtest_r2 - train_r2) / train_r2) * 100
    ax3.text(0.5, 0.95, f'Diferencia: {r2_diff:+.1f}%', 
             transform=ax3.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Gráfico guardado en: {output_path}")
    
    return output_path


@task(log_prints=True)
def plot_backtest_predictions(
    predictions_df: pd.DataFrame,
    sample_size: int = 500,
    output_path: str = None
) -> str:
    """
    Genera gráfico de predicciones vs valores reales del backtesting.
    """
    logger = get_run_logger()
    logger.info("📊 Generando gráfico de predicciones del backtesting...")
    
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(BACKTEST_DIR, f'backtest_predictions_{timestamp}.png')
    
    # Muestrear si hay muchos datos
    df = predictions_df.copy()
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Crear figura con 3 subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Análisis de Predicciones - Backtesting (Últimos 3 Meses)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Scatter: Predicciones vs Reales
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(df['Delayed_time'], df['Delayed_time_predicted'], 
                alpha=0.5, s=30, color='steelblue', edgecolors='black', linewidth=0.5)
    
    # Línea diagonal perfecta
    min_val = min(df['Delayed_time'].min(), df['Delayed_time_predicted'].min())
    max_val = max(df['Delayed_time'].max(), df['Delayed_time_predicted'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    
    ax1.set_xlabel('Delayed Time Real (días)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Delayed Time Predicho (días)', fontsize=11, fontweight='bold')
    ax1.set_title('Predicciones vs Valores Reales', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Agregar R²
    r2 = r2_score(df['Delayed_time'], df['Delayed_time_predicted'])
    ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11, fontweight='bold')
    
    # 2. Histograma de errores
    ax2 = fig.add_subplot(gs[1, 0])
    errors = df['prediction_error']
    ax2.hist(errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    ax2.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, 
                label=f'Media: {errors.mean():.2f}')
    ax2.set_xlabel('Error de Predicción (días)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax2.set_title('Distribución de Errores', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Boxplot de errores absolutos
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.boxplot([df['prediction_error_abs']], labels=['Backtest'], widths=0.6)
    ax3.set_ylabel('Error Absoluto (días)', fontsize=11, fontweight='bold')
    ax3.set_title('Distribución de Errores Absolutos', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Agregar estadísticas
    stats_text = f"Media: {df['prediction_error_abs'].mean():.2f}\n"
    stats_text += f"Mediana: {df['prediction_error_abs'].median():.2f}\n"
    stats_text += f"P95: {df['prediction_error_abs'].quantile(0.95):.2f}"
    ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, ha='right', va='top')
    
    # 4. Serie temporal (si existe timestamp)
    ax4 = fig.add_subplot(gs[2, :])
    if 'order_purchase_timestamp' in df.columns:
        df_sorted = df.sort_values('order_purchase_timestamp')
        ax4.plot(df_sorted['order_purchase_timestamp'], df_sorted['Delayed_time'], 
                label='Real', alpha=0.7, linewidth=1.5, color='steelblue')
        ax4.plot(df_sorted['order_purchase_timestamp'], df_sorted['Delayed_time_predicted'], 
                label='Predicho', alpha=0.7, linewidth=1.5, color='coral')
        ax4.set_xlabel('Fecha', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Delayed Time (días)', fontsize=11, fontweight='bold')
        ax4.set_title('Serie Temporal: Real vs Predicho', fontsize=13, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax4.text(0.5, 0.5, 'Timestamp no disponible', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Gráfico guardado en: {output_path}")
    
    return output_path


@task(log_prints=True)
def plot_error_distribution_by_category(
    predictions_df: pd.DataFrame,
    output_path: str = None
) -> str:
    """
    Genera gráfico de distribución de errores por categorías.
    """
    logger = get_run_logger()
    logger.info("📊 Generando gráfico de errores por categoría...")
    
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(BACKTEST_DIR, f'error_by_category_{timestamp}.png')
    
    df = predictions_df.copy()
    
    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis de Errores por Categorías - Backtesting', 
                 fontsize=16, fontweight='bold')
    
    # 1. Por rango de delay real
    ax1 = axes[0, 0]
    df['delay_range'] = pd.cut(df['Delayed_time'], 
                                bins=[-10, -1, 0, 3, 6, 20],
                                labels=['Muy Adelantado', 'Adelantado', 'A Tiempo', 
                                       'Poco Retrasado', 'Muy Retrasado'])
    
    delay_errors = df.groupby('delay_range')['prediction_error_abs'].agg(['mean', 'std', 'count'])
    delay_errors = delay_errors.sort_values('mean', ascending=False)
    
    ax1.barh(range(len(delay_errors)), delay_errors['mean'], color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(delay_errors)))
    ax1.set_yticklabels(delay_errors.index)
    ax1.set_xlabel('MAE (días)', fontsize=11, fontweight='bold')
    ax1.set_title('Error Promedio por Rango de Delay', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    for i, (idx, row) in enumerate(delay_errors.iterrows()):
        ax1.text(row['mean'], i, f" {row['mean']:.2f} (n={int(row['count'])})", 
                va='center', fontsize=9)
    
    # 2. Por magnitud del error
    ax2 = axes[0, 1]
    error_categories = ['< 1 día', '1-2 días', '2-3 días', '3-5 días', '> 5 días']
    error_counts = [
        (df['prediction_error_abs'] < 1).sum(),
        ((df['prediction_error_abs'] >= 1) & (df['prediction_error_abs'] < 2)).sum(),
        ((df['prediction_error_abs'] >= 2) & (df['prediction_error_abs'] < 3)).sum(),
        ((df['prediction_error_abs'] >= 3) & (df['prediction_error_abs'] < 5)).sum(),
        (df['prediction_error_abs'] >= 5).sum()
    ]
    error_pcts = [count / len(df) * 100 for count in error_counts]
    
    colors_pie = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#c0392b']
    wedges, texts, autotexts = ax2.pie(error_counts, labels=error_categories, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title('Distribución de Magnitud de Errores', fontsize=13, fontweight='bold')
    
    # 3. Top 10 peores predicciones
    ax3 = axes[1, 0]
    worst_10 = df.nlargest(10, 'prediction_error_abs')[['Delayed_time', 'Delayed_time_predicted', 'prediction_error_abs']]
    
    x = range(len(worst_10))
    width = 0.35
    ax3.bar([i - width/2 for i in x], worst_10['Delayed_time'], width, 
            label='Real', color='steelblue', alpha=0.7)
    ax3.bar([i + width/2 for i in x], worst_10['Delayed_time_predicted'], width, 
            label='Predicho', color='coral', alpha=0.7)
    
    ax3.set_xlabel('Top 10 Peores Predicciones', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Delayed Time (días)', fontsize=11, fontweight='bold')
    ax3.set_title('Top 10 Predicciones con Mayor Error', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'#{i+1}' for i in x])
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Estadísticas generales
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = "📊 ESTADÍSTICAS GENERALES\n\n"
    stats_text += f"Total predicciones: {len(df):,}\n\n"
    stats_text += f"MAE: {df['prediction_error_abs'].mean():.2f} días\n"
    stats_text += f"Mediana Error: {df['prediction_error_abs'].median():.2f} días\n"
    stats_text += f"Std Error: {df['prediction_error_abs'].std():.2f} días\n\n"
    stats_text += f"Percentiles:\n"
    stats_text += f"  • P25: {df['prediction_error_abs'].quantile(0.25):.2f} días\n"
    stats_text += f"  • P50: {df['prediction_error_abs'].quantile(0.50):.2f} días\n"
    stats_text += f"  • P75: {df['prediction_error_abs'].quantile(0.75):.2f} días\n"
    stats_text += f"  • P95: {df['prediction_error_abs'].quantile(0.95):.2f} días\n\n"
    stats_text += f"Predicciones excelentes (< 1 día): {error_pcts[0]:.1f}%\n"
    stats_text += f"Predicciones aceptables (< 3 días): {sum(error_pcts[:3]):.1f}%"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Gráfico guardado en: {output_path}")
    
    return output_path


@task(log_prints=True)
def generate_backtest_report(
    model,
    master_df: pd.DataFrame,
    feature_cols: List[str],
    train_metrics: Dict
) -> Dict:
    """
    Genera reporte completo de backtesting con dataset de últimos 3 meses.
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("📊 GENERANDO REPORTE COMPLETO DE BACKTESTING")
    logger.info("=" * 80)
    
    # 1. Cargar dataset de últimos 3 meses
    backtest_df = load_last_3_months_data()
    
    # 2. Realizar validación
    backtest_results = perform_backtest_validation(
        model=model,
        backtest_df=backtest_df,
        master_df=master_df,
        feature_cols=feature_cols
    )
    
    if backtest_results is None:
        logger.error("   ❌ Error en backtesting, abortando generación de gráficos")
        return None
    
    # 3. Generar gráficos
    viz_paths = {}
    
    # Gráfico de comparación Train vs Backtest
    comparison_plot = plot_backtest_comparison(backtest_results, train_metrics)
    viz_paths['train_vs_backtest'] = comparison_plot
    
    # Gráfico de predicciones
    predictions_plot = plot_backtest_predictions(backtest_results['predictions_df'])
    viz_paths['backtest_predictions'] = predictions_plot
    
    # Gráfico de errores por categoría
    error_plot = plot_error_distribution_by_category(backtest_results['predictions_df'])
    viz_paths['error_by_category'] = error_plot
    
    logger.info("=" * 80)
    logger.info("✅ REPORTE DE BACKTESTING COMPLETADO")
    logger.info("=" * 80)
    for viz_name, viz_path in viz_paths.items():
        logger.info(f"   📊 {viz_name}: {viz_path}")
    logger.info("=" * 80)
    
    return {
        'backtest_results': backtest_results,
        'visualizations': viz_paths,
        'metrics': backtest_results['metrics']
    }