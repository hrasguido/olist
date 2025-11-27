"""
Evaluación con validación cruzada y comparación de modelos.
Compara múltiples algoritmos usando cross-validation y métricas estadísticas.
"""
from prefect import task, flow, get_run_logger
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
#from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
import os
warnings.filterwarnings('ignore')

# Configurar carpeta de outputs (ruta relativa)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)



@task(log_prints=True)
def prepare_data_for_cv(
    master_df: pd.DataFrame, 
    target_col: str = 'Delayed_time'
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepara los datos para validación cruzada.
    
    Args:
        master_df: DataFrame con features y target
        target_col: Nombre de la columna target
    
    Returns:
        Tuple con (X, y, feature_names)
    """
    logger = get_run_logger()
    logger.info("📊 Preparando datos para validación cruzada...")
    
    # Identificar columnas
    id_columns = [col for col in master_df.columns if 'id' in col.lower()]
    date_columns = [col for col in master_df.columns if master_df[col].dtype == 'datetime64[ns]']
    
    # Features numéricas (excluyendo IDs, fechas, target y predicciones previas)
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols 
                   if col != target_col 
                   and col not in id_columns
                   and 'predicted' not in col.lower()
                   and 'error' not in col.lower()]
    
    logger.info(f"   • Features seleccionadas: {len(feature_cols)}")
    logger.info(f"   • Registros totales: {len(master_df):,}")
    
    # Preparar X e y
    X = master_df[feature_cols].fillna(0)
    y = master_df[target_col].fillna(0)
    
    return X, y, feature_cols


@task(log_prints=True)
def cross_validate_model(
    model,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Evalúa un modelo usando validación cruzada.
    
    Args:
        model: Modelo de sklearn/xgboost
        model_name: Nombre del modelo
        X: Features
        y: Target
        cv_folds: Número de folds para CV
        random_state: Semilla aleatoria
    
    Returns:
        Dict con métricas de CV
    """
    logger = get_run_logger()
    logger.info(f"   🔄 Evaluando {model_name} con {cv_folds}-Fold CV...")
    
    # Definir métricas
    scoring = {
        'neg_mae': 'neg_mean_absolute_error',
        'neg_rmse': 'neg_root_mean_squared_error',
        'r2': 'r2'
    }
    
    # K-Fold cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Ejecutar cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=kf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Calcular estadísticas
    results = {
        'model_name': model_name,
        'cv_folds': cv_folds,
        'train_mae_mean': -cv_results['train_neg_mae'].mean(),
        'train_mae_std': cv_results['train_neg_mae'].std(),
        'test_mae_mean': -cv_results['test_neg_mae'].mean(),
        'test_mae_std': cv_results['test_neg_mae'].std(),
        'train_rmse_mean': -cv_results['train_neg_rmse'].mean(),
        'train_rmse_std': cv_results['train_neg_rmse'].std(),
        'test_rmse_mean': -cv_results['test_neg_rmse'].mean(),
        'test_rmse_std': cv_results['test_neg_rmse'].std(),
        'train_r2_mean': cv_results['train_r2'].mean(),
        'train_r2_std': cv_results['train_r2'].std(),
        'test_r2_mean': cv_results['test_r2'].mean(),
        'test_r2_std': cv_results['test_r2'].std(),
        'fit_time_mean': cv_results['fit_time'].mean(),
        'fit_time_std': cv_results['fit_time'].std(),
    }
    
    logger.info(f"      ✅ Test MAE: {results['test_mae_mean']:.3f} ± {results['test_mae_std']:.3f}")
    logger.info(f"      ✅ Test RMSE: {results['test_rmse_mean']:.3f} ± {results['test_rmse_std']:.3f}")
    logger.info(f"      ✅ Test R²: {results['test_r2_mean']:.4f} ± {results['test_r2_std']:.4f}")
    logger.info(f"      ⏱️  Tiempo: {results['fit_time_mean']:.2f}s ± {results['fit_time_std']:.2f}s")
    
    return results


@task(log_prints=True)
def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compara múltiples modelos usando validación cruzada.
    
    Args:
        X: Features
        y: Target
        cv_folds: Número de folds para CV
        random_state: Semilla aleatoria
    
    Returns:
        DataFrame con resultados comparativos
    """
    logger = get_run_logger()
    logger.info("=" * 80)
    logger.info("🏆 COMPARACIÓN DE MODELOS CON VALIDACIÓN CRUZADA")
    logger.info("=" * 80)
    logger.info(f"   • Método: {cv_folds}-Fold Cross-Validation")
    logger.info(f"   • Registros: {len(X):,}")
    logger.info(f"   • Features: {len(X.columns)}")
    logger.info("")
    
    # Definir modelos a comparar
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=random_state),
        'Lasso Regression': Lasso(alpha=1.0, random_state=random_state),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
    }
    
    # Evaluar cada modelo
    all_results = []
    for model_name, model in models.items():
        try:
            results = cross_validate_model(
                model, model_name, X, y, cv_folds, random_state
            )
            all_results.append(results)
        except Exception as e:
            logger.warning(f"   ⚠️  Error evaluando {model_name}: {str(e)}")
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame(all_results)
    
    # Ordenar por Test MAE (menor es mejor)
    results_df = results_df.sort_values('test_mae_mean')
    
    return results_df


@task(log_prints=True)
def statistical_comparison(results_df: pd.DataFrame) -> Dict:
    """
    Realiza comparación estadística entre modelos.
    
    Args:
        results_df: DataFrame con resultados de CV
    
    Returns:
        Dict con análisis estadístico
    """
    logger = get_run_logger()
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 ANÁLISIS ESTADÍSTICO DE MODELOS")
    logger.info("=" * 80)
    
    # Mejor modelo por cada métrica
    best_mae = results_df.loc[results_df['test_mae_mean'].idxmin()]
    best_rmse = results_df.loc[results_df['test_rmse_mean'].idxmin()]
    best_r2 = results_df.loc[results_df['test_r2_mean'].idxmax()]
    best_time = results_df.loc[results_df['fit_time_mean'].idxmin()]
    
    logger.info("")
    logger.info("🏆 MEJORES MODELOS POR MÉTRICA:")
    logger.info(f"   • Mejor MAE: {best_mae['model_name']} ({best_mae['test_mae_mean']:.3f} ± {best_mae['test_mae_std']:.3f})")
    logger.info(f"   • Mejor RMSE: {best_rmse['model_name']} ({best_rmse['test_rmse_mean']:.3f} ± {best_rmse['test_rmse_std']:.3f})")
    logger.info(f"   • Mejor R²: {best_r2['model_name']} ({best_r2['test_r2_mean']:.4f} ± {best_r2['test_r2_std']:.4f})")
    logger.info(f"   • Más rápido: {best_time['model_name']} ({best_time['fit_time_mean']:.2f}s ± {best_time['fit_time_std']:.2f}s)")
    
    # Ranking general (promedio de rankings normalizados)
    logger.info("")
    logger.info("📈 RANKING GENERAL:")
    
    # Normalizar métricas (0-1, donde 1 es mejor)
    results_df['mae_rank'] = 1 - (results_df['test_mae_mean'] - results_df['test_mae_mean'].min()) / (results_df['test_mae_mean'].max() - results_df['test_mae_mean'].min() + 1e-10)
    results_df['rmse_rank'] = 1 - (results_df['test_rmse_mean'] - results_df['test_rmse_mean'].min()) / (results_df['test_rmse_mean'].max() - results_df['test_rmse_mean'].min() + 1e-10)
    results_df['r2_rank'] = (results_df['test_r2_mean'] - results_df['test_r2_mean'].min()) / (results_df['test_r2_mean'].max() - results_df['test_r2_mean'].min() + 1e-10)
    
    # Score general (promedio de rankings)
    results_df['overall_score'] = (results_df['mae_rank'] + results_df['rmse_rank'] + results_df['r2_rank']) / 3
    results_df_sorted = results_df.sort_values('overall_score', ascending=False)
    
    for idx, (_, row) in enumerate(results_df_sorted.iterrows(), 1):
        logger.info(f"   {idx}. {row['model_name']}: {row['overall_score']:.4f}")
    
    # Análisis de overfitting
    logger.info("")
    logger.info("🔍 ANÁLISIS DE OVERFITTING (Train vs Test):")
    for _, row in results_df.iterrows():
        train_test_gap_mae = row['train_mae_mean'] - row['test_mae_mean']
        train_test_gap_r2 = row['train_r2_mean'] - row['test_r2_mean']
        
        overfitting_status = "✅ Bueno" if abs(train_test_gap_r2) < 0.1 else "⚠️  Overfitting"
        
        logger.info(f"   • {row['model_name']}:")
        logger.info(f"      - Gap MAE: {train_test_gap_mae:.3f} días")
        logger.info(f"      - Gap R²: {train_test_gap_r2:.4f} {overfitting_status}")
    
    return {
        'best_mae_model': best_mae['model_name'],
        'best_rmse_model': best_rmse['model_name'],
        'best_r2_model': best_r2['model_name'],
        'best_overall_model': results_df_sorted.iloc[0]['model_name'],
        'results_df': results_df_sorted
    }


@task(log_prints=True)
def print_comparison_table(results_df: pd.DataFrame):
    """
    Imprime tabla comparativa formateada de resultados.
    
    Args:
        results_df: DataFrame con resultados de CV
    """
    logger = get_run_logger()
    logger.info("")
    logger.info("=" * 80)
    logger.info("📋 TABLA COMPARATIVA DE MODELOS")
    logger.info("=" * 80)
    logger.info("")
    
    # Crear tabla formateada
    logger.info(f"{'Modelo':<25} {'Test MAE':<18} {'Test RMSE':<18} {'Test R²':<18} {'Tiempo (s)':<12}")
    logger.info("-" * 95)
    
    for _, row in results_df.iterrows():
        mae_str = f"{row['test_mae_mean']:.3f} ± {row['test_mae_std']:.3f}"
        rmse_str = f"{row['test_rmse_mean']:.3f} ± {row['test_rmse_std']:.3f}"
        r2_str = f"{row['test_r2_mean']:.4f} ± {row['test_r2_std']:.4f}"
        time_str = f"{row['fit_time_mean']:.2f} ± {row['fit_time_std']:.2f}"
        
        logger.info(f"{row['model_name']:<25} {mae_str:<18} {rmse_str:<18} {r2_str:<18} {time_str:<12}")
    
    logger.info("=" * 80)


@task(log_prints=True)
def save_comparison_results(
    results_df: pd.DataFrame,
    output_path: str = None
):
    """
    Guarda los resultados de comparación en un archivo CSV.
    
    Args:
        results_df: DataFrame con resultados
        output_path: Ruta del archivo de salida
    """
    logger = get_run_logger()
    
    # Definir ruta por defecto con timestamp
    if output_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(OUTPUT_DIR, f'model_comparison_{timestamp}.csv')
    
    logger.info(f"💾 Guardando resultados en: {output_path}")
    
    # Seleccionar columnas relevantes
    output_cols = [
        'model_name', 
        'test_mae_mean', 'test_mae_std',
        'test_rmse_mean', 'test_rmse_std',
        'test_r2_mean', 'test_r2_std',
        'train_mae_mean', 'train_r2_mean',
        'fit_time_mean'
    ]
    
    # Agregar overall_score si existe
    if 'overall_score' in results_df.columns:
        output_cols.append('overall_score')
    
    results_df[output_cols].to_csv(output_path, index=False)
    logger.info(f"   ✅ Resultados guardados exitosamente")
    
    return output_path


@flow(name="Model Evaluation with Cross-Validation", log_prints=True)
def evaluate_models_with_cv(
    master_df: pd.DataFrame,
    target_col: str = 'Delayed_time',
    cv_folds: int = 5,
    save_results: bool = True
) -> Dict:
    """
    Función principal para evaluación con validación cruzada y comparación de modelos.
    
    Args:
        master_df: DataFrame con features y target
        target_col: Nombre de la columna target
        cv_folds: Número de folds para CV
        save_results: Si True, guarda resultados en CSV
    
    Returns:
        Dict con resultados completos
    """
    logger = get_run_logger()
    
    # 1. Preparar datos
    X, y, feature_cols = prepare_data_for_cv(master_df, target_col)
    
    # 2. Comparar modelos
    results_df = compare_models(X, y, cv_folds)
    
    # 3. Análisis estadístico
    stats_results = statistical_comparison(results_df)
    
    # 4. Imprimir tabla comparativa
    print_comparison_table(stats_results['results_df'])
    
    # 5. Guardar resultados (opcional)
    if save_results:
        comparison_path = save_comparison_results(stats_results['results_df'])
    
    return {
        'results_df': stats_results['results_df'],
        'best_model': stats_results['best_overall_model'],
        'statistics': stats_results,
        'n_features': len(feature_cols),
        'n_samples': len(X),
        'comparison_path': comparison_path if save_results else None
    }

if __name__ == "__main__":
    print("Modulo de evaluacion")