# src/visualization.py
"""
Módulo para generación de visualizaciones del modelo.
Genera gráficos de rendimiento, feature importance y análisis de errores.
"""

from prefect import task, get_run_logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Configurar carpeta de outputs (ruta relativa)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)


@task(log_prints=True)
def plot_feature_importance(
    model,
    feature_cols: List[str],
    top_n: int = 20,
    output_path: str = None
) -> str:
    """
    Genera gráfico de importancia de features.
    
    Args:
        model: Modelo entrenado con feature_importances_
        feature_cols: Lista de nombres de features
        top_n: Número de features más importantes a mostrar
        output_path: Ruta del archivo de salida
    
    Returns:
        Ruta del archivo guardado
    """
    logger = get_run_logger()
    logger.info(f"📊 Generando gráfico de Feature Importance (Top {top_n})...")
    
    # Definir ruta por defecto con timestamp
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(VISUALIZATIONS_DIR, f'feature_importance_plot_{timestamp}.png')
    
    # Crear DataFrame con importancias
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Gráfico de barras horizontal
    colors = sns.color_palette("viridis", len(feature_importance))
    bars = ax.barh(range(len(feature_importance)), feature_importance['importance'], color=colors)
    
    # Configurar ejes
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importancia', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Features Más Importantes - XGBoost Model', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Agregar valores en las barras
    for i, (idx, row) in enumerate(feature_importance.iterrows()):
        ax.text(row['importance'], i, f" {row['importance']:.4f}", 
                va='center', fontsize=9)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Gráfico guardado en: {output_path}")
    
    return output_path


@task(log_prints=True)
def plot_model_performance(
    master_df: pd.DataFrame,
    model_metrics: Dict,
    output_path: str = None
) -> str:
    """
    Genera gráfico de rendimiento del modelo con múltiples subplots.
    
    Args:
        master_df: DataFrame con predicciones y valores reales
        model_metrics: Diccionario con métricas del modelo
        output_path: Ruta del archivo de salida
    
    Returns:
        Ruta del archivo guardado
    """
    logger = get_run_logger()
    logger.info("📊 Generando gráfico de rendimiento del modelo...")
    
    # Definir ruta por defecto con timestamp
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(VISUALIZATIONS_DIR, f'model_performance_plot_{timestamp}.png')
    
    # Verificar columnas necesarias
    if 'Delayed_time' not in master_df.columns or 'Delayed_time_predicted' not in master_df.columns:
        logger.warning("⚠️  Columnas necesarias no encontradas para el gráfico de rendimiento")
        return None
    
    # Crear figura con subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # ============================================================
    # 1. SCATTER PLOT: Predicciones vs Valores Reales
    # ============================================================
    ax1 = fig.add_subplot(gs[0, :])
    
    # Limitar a un sample para mejor visualización
    sample_size = min(5000, len(master_df))
    sample_df = master_df.sample(n=sample_size, random_state=42)
    
    ax1.scatter(sample_df['Delayed_time'], sample_df['Delayed_time_predicted'], 
                alpha=0.5, s=20, c='steelblue', edgecolors='none')
    
    # Línea de predicción perfecta
    max_val = max(sample_df['Delayed_time'].max(), sample_df['Delayed_time_predicted'].max())
    min_val = min(sample_df['Delayed_time'].min(), sample_df['Delayed_time_predicted'].min())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
    
    ax1.set_xlabel('Delayed Time Real (días)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Delayed Time Predicho (días)', fontsize=11, fontweight='bold')
    ax1.set_title('Predicciones vs Valores Reales', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Agregar métricas en el gráfico
    test_metrics = model_metrics.get('test', {})
    metrics_text = f"MAE: {test_metrics.get('mae', 0):.2f} días\n"
    metrics_text += f"RMSE: {test_metrics.get('rmse', 0):.2f} días\n"
    metrics_text += f"R²: {test_metrics.get('r2', 0):.4f}"
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ============================================================
    # 2. DISTRIBUCIÓN DE ERRORES
    # ============================================================
    ax2 = fig.add_subplot(gs[1, 0])
    
    errors = master_df['prediction_error']
    ax2.hist(errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {errors.mean():.2f}')
    ax2.axvline(errors.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {errors.median():.2f}')
    
    ax2.set_xlabel('Error de Predicción (días)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax2.set_title('Distribución de Errores de Predicción', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ============================================================
    # 3. DISTRIBUCIÓN DE ERRORES ABSOLUTOS
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 1])
    
    abs_errors = master_df['prediction_error_abs']
    ax3.hist(abs_errors, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax3.axvline(abs_errors.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'MAE: {abs_errors.mean():.2f}')
    ax3.axvline(abs_errors.median(), color='green', linestyle='--', linewidth=2, 
                label=f'Mediana: {abs_errors.median():.2f}')
    
    ax3.set_xlabel('Error Absoluto (días)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax3.set_title('Distribución de Errores Absolutos', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ============================================================
    # 4. RESIDUOS vs PREDICCIONES
    # ============================================================
    ax4 = fig.add_subplot(gs[2, 0])
    
    ax4.scatter(sample_df['Delayed_time_predicted'], sample_df['prediction_error'], 
                alpha=0.5, s=20, c='purple', edgecolors='none')
    ax4.axhline(0, color='red', linestyle='--', linewidth=2)
    
    ax4.set_xlabel('Delayed Time Predicho (días)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Residuos (días)', fontsize=11, fontweight='bold')
    ax4.set_title('Residuos vs Predicciones', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ============================================================
    # 5. MÉTRICAS POR CONJUNTO (Train/Val/Test)
    # ============================================================
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Preparar datos de métricas
    sets = ['Train', 'Val', 'Test']
    mae_values = [
        model_metrics.get('train', {}).get('mae', 0),
        model_metrics.get('val', {}).get('mae', 0),
        model_metrics.get('test', {}).get('mae', 0)
    ]
    r2_values = [
        model_metrics.get('train', {}).get('r2', 0),
        model_metrics.get('val', {}).get('r2', 0),
        model_metrics.get('test', {}).get('r2', 0)
    ]
    
    x = np.arange(len(sets))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, mae_values, width, label='MAE', color='coral', alpha=0.8)
    ax5_twin = ax5.twinx()
    bars2 = ax5_twin.bar(x + width/2, r2_values, width, label='R²', color='steelblue', alpha=0.8)
    
    ax5.set_xlabel('Conjunto de Datos', fontsize=11, fontweight='bold')
    ax5.set_ylabel('MAE (días)', fontsize=11, fontweight='bold', color='coral')
    ax5_twin.set_ylabel('R²', fontsize=11, fontweight='bold', color='steelblue')
    ax5.set_title('Métricas por Conjunto de Datos', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(sets)
    ax5.tick_params(axis='y', labelcolor='coral')
    ax5_twin.tick_params(axis='y', labelcolor='steelblue')
    
    # Agregar valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax5_twin.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Título general
    fig.suptitle('Análisis de Rendimiento del Modelo XGBoost', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Guardar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Gráfico guardado en: {output_path}")
    
    return output_path


@task(log_prints=True)
def plot_cv_comparison(
    cv_results_df: pd.DataFrame,
    output_path: str = None
) -> str:
    """
    Genera gráfico de comparación de modelos con cross-validation.
    
    Args:
        cv_results_df: DataFrame con resultados de CV
        output_path: Ruta del archivo de salida
    
    Returns:
        Ruta del archivo guardado
    """
    logger = get_run_logger()
    logger.info("📊 Generando gráfico de comparación de modelos...")
    
    # Definir ruta por defecto con timestamp
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(VISUALIZATIONS_DIR, f'model_comparison_plot_{timestamp}.png')
    
    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparación de Modelos - Cross Validation', 
                 fontsize=16, fontweight='bold')
    
    # Preparar datos
    models = cv_results_df['model_name'].values
    x = np.arange(len(models))
    
    # ============================================================
    # 1. MAE Comparison
    # ============================================================
    ax1 = axes[0, 0]
    mae_mean = cv_results_df['test_mae_mean'].values
    mae_std = cv_results_df['test_mae_std'].values
    
    bars = ax1.bar(x, mae_mean, yerr=mae_std, capsize=5, 
                   color=sns.color_palette("Set2", len(models)), alpha=0.8)
    ax1.set_xlabel('Modelo', fontsize=11, fontweight='bold')
    ax1.set_ylabel('MAE (días)', fontsize=11, fontweight='bold')
    ax1.set_title('Mean Absolute Error (MAE)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae_mean[i]:.2f}±{mae_std[i]:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # ============================================================
    # 2. R² Comparison
    # ============================================================
    ax2 = axes[0, 1]
    r2_mean = cv_results_df['test_r2_mean'].values
    r2_std = cv_results_df['test_r2_std'].values
    
    bars = ax2.bar(x, r2_mean, yerr=r2_std, capsize=5,
                   color=sns.color_palette("Set3", len(models)), alpha=0.8)
    ax2.set_xlabel('Modelo', fontsize=11, fontweight='bold')
    ax2.set_ylabel('R²', fontsize=11, fontweight='bold')
    ax2.set_title('Coeficiente de Determinación (R²)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2_mean[i]:.3f}±{r2_std[i]:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # ============================================================
    # 3. RMSE Comparison
    # ============================================================
    ax3 = axes[1, 0]
    rmse_mean = cv_results_df['test_rmse_mean'].values
    rmse_std = cv_results_df['test_rmse_std'].values
    
    bars = ax3.bar(x, rmse_mean, yerr=rmse_std, capsize=5,
                   color=sns.color_palette("Pastel1", len(models)), alpha=0.8)
    ax3.set_xlabel('Modelo', fontsize=11, fontweight='bold')
    ax3.set_ylabel('RMSE (días)', fontsize=11, fontweight='bold')
    ax3.set_title('Root Mean Squared Error (RMSE)', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse_mean[i]:.2f}±{rmse_std[i]:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # ============================================================
    # 4. Training Time Comparison
    # ============================================================
    ax4 = axes[1, 1]
    fit_time = cv_results_df['fit_time_mean'].values
    
    bars = ax4.bar(x, fit_time, color=sns.color_palette("husl", len(models)), alpha=0.8)
    ax4.set_xlabel('Modelo', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Tiempo (segundos)', fontsize=11, fontweight='bold')
    ax4.set_title('Tiempo de Entrenamiento', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{fit_time[i]:.2f}s',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Guardar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Gráfico guardado en: {output_path}")
    
    return output_path


@task(log_prints=True)
def generate_all_visualizations(
    master_df: pd.DataFrame,
    model,
    feature_cols: List[str],
    model_metrics: Dict,
    cv_results: Dict = None
) -> Dict[str, str]:
    """
    Genera todas las visualizaciones del modelo.
    
    Args:
        master_df: DataFrame con predicciones
        model: Modelo entrenado
        feature_cols: Lista de features
        model_metrics: Métricas del modelo
        cv_results: Resultados de cross-validation
    
    Returns:
        Dict con rutas de todos los gráficos generados
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("📊 GENERANDO VISUALIZACIONES")
    logger.info("=" * 80)
    
    viz_paths = {}
    
    # 1. Feature Importance
    feature_importance_path = plot_feature_importance(model, feature_cols, top_n=20)
    viz_paths['feature_importance'] = feature_importance_path
    
    # 2. Model Performance
    performance_path = plot_model_performance(master_df, model_metrics)
    viz_paths['model_performance'] = performance_path
    
    # 3. CV Comparison (si está disponible)
    if cv_results and 'results_df' in cv_results:
        cv_comparison_path = plot_cv_comparison(cv_results['results_df'])
        viz_paths['cv_comparison'] = cv_comparison_path
    
    logger.info("=" * 80)
    logger.info("✅ VISUALIZACIONES COMPLETADAS")
    logger.info("=" * 80)
    for viz_name, viz_path in viz_paths.items():
        logger.info(f"   📊 {viz_name}: {viz_path}")
    logger.info("=" * 80)

    # 4. Optuna vs Baseline (si está disponible)
    if cv_results and 'results_df' in cv_results:
        optuna_comparison_path = plot_optuna_vs_baseline(
            cv_results['results_df'], 
            model_metrics
        )
        viz_paths['optuna_comparison'] = optuna_comparison_path
    
    return viz_paths

@task(log_prints=True)
def plot_optuna_vs_baseline(
    cv_results_df: pd.DataFrame,
    optuna_metrics: Dict,
    output_path: str = None
) -> str:
    """
    Compara el modelo optimizado con Optuna vs los modelos baseline.
    """
    logger = get_run_logger()
    logger.info("📊 Generando gráfico Optuna vs Baseline...")
    
    # Definir ruta
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(VISUALIZATIONS_DIR, f'optuna_comparison_{timestamp}.png')
    
    # Crear figura
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('XGBoost: Baseline vs Optimizado con Optuna', 
                 fontsize=16, fontweight='bold')
    
    # Obtener métricas de XGBoost baseline
    xgb_baseline = cv_results_df[cv_results_df['model_name'] == 'XGBoost'].iloc[0]
    
    # Datos para comparación
    models = ['XGBoost\n(Baseline)', 'XGBoost\n(Optuna)']
    mae_values = [xgb_baseline['test_mae_mean'], optuna_metrics['test']['mae']]
    rmse_values = [xgb_baseline['test_rmse_mean'], optuna_metrics['test']['rmse']]
    r2_values = [xgb_baseline['test_r2_mean'], optuna_metrics['test']['r2']]
    
    # MAE
    ax1 = axes[0]
    bars = ax1.bar(models, mae_values, color=['steelblue', 'coral'], alpha=0.8, edgecolor='none')
    ax1.set_ylabel('MAE (días)', fontsize=11, fontweight='bold')
    ax1.set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae_values[i]:.2f}', ha='center', va='bottom', fontsize=10)
    
    # RMSE
    ax2 = axes[1]
    bars = ax2.bar(models, rmse_values, color=['steelblue', 'coral'], alpha=0.8, edgecolor='none')
    ax2.set_ylabel('RMSE (días)', fontsize=11, fontweight='bold')
    ax2.set_title('Root Mean Squared Error', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{rmse_values[i]:.2f}', ha='center', va='bottom', fontsize=10)
    
    # R²
    ax3 = axes[2]
    bars = ax3.bar(models, r2_values, color=['steelblue', 'coral'], alpha=0.8, edgecolor='none')
    ax3.set_ylabel('R²', fontsize=11, fontweight='bold')
    ax3.set_title('Coeficiente de Determinación', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{r2_values[i]:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Calcular mejora
    mae_improvement = ((xgb_baseline['test_mae_mean'] - optuna_metrics['test']['mae']) / 
                       xgb_baseline['test_mae_mean'] * 100)
    r2_improvement = ((optuna_metrics['test']['r2'] - xgb_baseline['test_r2_mean']) / 
                      xgb_baseline['test_r2_mean'] * 100)
    
    # Agregar texto de mejora
    fig.text(0.5, 0.02, 
             f"Mejora con Optuna: MAE {mae_improvement:+.1f}% | R² {r2_improvement:+.1f}%",
             ha='center', fontsize=12, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Gráfico guardado en: {output_path}")
    logger.info(f"   📈 Mejora MAE: {mae_improvement:+.1f}%")
    logger.info(f"   📈 Mejora R²: {r2_improvement:+.1f}%")
    
    return output_path