# src/models/visualize.py
"""Módulo para visualizaciones de modelos."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os


def plot_regression_analysis(results, output_dir='outputs/'):
    """Genera gráficos de análisis de regresión (scatter + residuos).
    
    Args:
        results: dict con resultados de regresión
        output_dir: Directorio de salida
    """
    y_test = results['y_test']
    y_pred = results['y_pred_best']
    best_model = results['best_metrics']['model_name']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Scatter plot
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.5, s=20)
    ax1.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Predicción perfecta')
    ax1.set_xlabel('Delay Days Real', fontsize=12)
    ax1.set_ylabel('Delay Days Predicho', fontsize=12)
    ax1.set_title(f'Predicciones vs Valores Reales\n({best_model})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Residuos
    ax2 = axes[1]
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Delay Days Predicho', fontsize=12)
    ax2.set_ylabel('Residuos', fontsize=12)
    ax2.set_title('Análisis de Residuos', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'regression_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n✅ Gráficos de regresión guardados en: {filepath}")


def plot_model_comparison(results, output_dir='outputs/'):
    """Genera gráficos comparativos de métricas.
    
    Args:
        results: dict con resultados de regresión
        output_dir: Directorio de salida
    """
    results_df = results['results_df']
    best_model = results['best_metrics']['model_name']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Comparación de RMSE
    ax1 = axes[0, 0]
    colors = ['#2ecc71' if m == best_model else '#3498db' for m in results_df['Modelo']]
    ax1.barh(results_df['Modelo'], results_df['RMSE'], color=colors)
    ax1.set_xlabel('RMSE (días)', fontsize=11, fontweight='bold')
    ax1.set_title('Comparación de RMSE por Modelo', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    for i, v in enumerate(results_df['RMSE']):
        ax1.text(v + 0.1, i, f'{v:.3f}', va='center', fontsize=9)
    
    # 2. Comparación de MAE
    ax2 = axes[0, 1]
    colors = ['#2ecc71' if m == best_model else '#e74c3c' for m in results_df['Modelo']]
    ax2.barh(results_df['Modelo'], results_df['MAE'], color=colors)
    ax2.set_xlabel('MAE (días)', fontsize=11, fontweight='bold')
    ax2.set_title('Comparación de MAE por Modelo', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    for i, v in enumerate(results_df['MAE']):
        ax2.text(v + 0.1, i, f'{v:.3f}', va='center', fontsize=9)
    
    # 3. Comparación de R²
    ax3 = axes[1, 0]
    colors = ['#2ecc71' if m == best_model else '#9b59b6' for m in results_df['Modelo']]
    ax3.barh(results_df['Modelo'], results_df['R²'], color=colors)
    ax3.set_xlabel('R² Score', fontsize=11, fontweight='bold')
    ax3.set_title('Comparación de R² por Modelo', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    for i, v in enumerate(results_df['R²']):
        ax3.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    # 4. Tabla resumen
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for _, row in results_df.iterrows():
        table_data.append([
            row['Modelo'],
            f"{row['RMSE']:.3f}",
            f"{row['MAE']:.3f}",
            f"{row['R²']:.3f}"
        ])
    
    table = ax4.table(
        cellText=table_data,
        colLabels=['Modelo', 'RMSE', 'MAE', 'R²'],
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Destacar mejor modelo
    for i, (_, row) in enumerate(results_df.iterrows()):
        if row['Modelo'] == best_model:
            for j in range(4):
                table[(i+1, j)].set_facecolor('#2ecc71')
                table[(i+1, j)].set_text_props(weight='bold')
    
    # Estilo de encabezados
    for j in range(4):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Resumen de Métricas', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Gráficos comparativos guardados en: {filepath}")


def plot_predictions_comparison(results, output_dir='outputs/'):
    """Genera grid de comparación visual de predicciones de todos los modelos.
    
    Args:
        results: dict con resultados de regresión
        output_dir: Directorio de salida
    """
    predictions = results['predictions']
    y_test = results['y_test']
    best_model = results['best_metrics']['model_name']
    
    n_models = len(predictions)
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, y_pred_model) in enumerate(predictions.items()):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(y_test, y_pred_model, alpha=0.4, s=15)
        ax.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, alpha=0.7)
        
        # Calcular métricas para este modelo
        model_rmse = np.sqrt(mean_squared_error(y_test, y_pred_model))
        model_r2 = r2_score(y_test, y_pred_model)
        
        # Título con métricas
        title_color = '#2ecc71' if model_name == best_model else '#34495e'
        title_weight = 'bold' if model_name == best_model else 'normal'
        ax.set_title(f'{model_name}\nRMSE: {model_rmse:.3f} | R²: {model_r2:.3f}', 
                    fontsize=10, fontweight=title_weight, color=title_color)
        
        ax.set_xlabel('Delay Days Real', fontsize=9)
        ax.set_ylabel('Delay Days Predicho', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Destacar mejor modelo con borde
        if model_name == best_model:
            for spine in ax.spines.values():
                spine.set_edgecolor('#2ecc71')
                spine.set_linewidth(3)
    
    # Ocultar ejes sobrantes
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Comparación Visual de Predicciones - Todos los Modelos', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'predictions_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Comparación de predicciones guardada en: {filepath}")