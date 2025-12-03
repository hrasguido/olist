# save_model/load.py
"""
Script para cargar el modelo entrenado y realizar predicciones.
Utiliza el modelo XGBoost más reciente del proyecto Olist.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def get_latest_model_path(models_dir: str = None) -> str:
    """
    Obtiene la ruta del modelo más reciente.
    
    Args:
        models_dir: Directorio donde están los modelos
    
    Returns:
        Ruta del modelo más reciente
    """
    if models_dir is None:
        # Ruta relativa desde save_model/ hacia outputs/pkls/
        base_dir = Path(__file__).parent.parent
        models_dir = base_dir / 'outputs' / 'pkls'
    else:
        models_dir = Path(models_dir)
    
    # Buscar todos los archivos .pkl
    model_files = list(models_dir.glob('xgboost_model_*.pkl'))
    
    if not model_files:
        raise FileNotFoundError(f"No se encontraron modelos en {models_dir}")
    
    # Ordenar por fecha de modificación (más reciente primero)
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    print(f"📦 Modelo más reciente: {latest_model.name}")
    print(f"📅 Fecha: {pd.Timestamp.fromtimestamp(latest_model.stat().st_mtime)}")
    
    return str(latest_model)


def load_model(model_path: str = None) -> Dict:
    """
    Carga el modelo entrenado desde pickle.
    
    Args:
        model_path: Ruta del modelo (si None, carga el más reciente)
    
    Returns:
        Dict con modelo, features y metadatos
    """
    if model_path is None:
        model_path = get_latest_model_path()
    
    print(f"\n🔄 Cargando modelo desde: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_package = pickle.load(f)
    
    # Extraer componentes
    model = model_package['model']
    features = model_package['feature_columns']
    metrics = model_package.get('model_metrics', {})
    
    print(f"✅ Modelo cargado exitosamente")
    
    if metrics and 'test' in metrics:
        print(f"   • Test MAE: {metrics['test']['mae']:.2f} días")
        print(f"   • Test R²: {metrics['test']['r2']:.4f}")
    
    return {
        'model': model,
        'features': features,
        'metrics': metrics,
        'timestamp': model_package.get('timestamp', 'N/A')
    }


def prepare_features(data: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    """
    Prepara las features del dataset para predicción.
    
    Args:
        data: DataFrame con datos crudos
        required_features: Lista de features requeridas por el modelo
    
    Returns:
        DataFrame con features preparadas
    """
    print(f"\n🔧 Preparando features...")
    print(f"   • Registros de entrada: {len(data):,}")
    print(f"   • Columnas disponibles: {len(data.columns)}")
    
    # Crear copia
    df = data.copy()
    
    # Identificar features faltantes
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        # Rellenar features faltantes con 0
        for feature in missing_features:
            df[feature] = 0
    
    # Seleccionar solo las features necesarias
    X = df[required_features].copy()
    
    # Rellenar NaN con 0
    X = X.fillna(0)
    
    # Reemplazar infinitos con 0
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"   ✅ Features preparadas: {X.shape[1]} columnas, {X.shape[0]} filas")
    
    return X


def predict_delivery_delay(
    order_data: pd.DataFrame,
    model_path: str = None,
    include_probabilities: bool = False
) -> pd.DataFrame:
    """
    Predice el retraso en días para nuevos pedidos.
    
    Args:
        order_data: DataFrame con datos del pedido
        model_path: Ruta del modelo (si None, usa el más reciente)
        include_probabilities: Si True, incluye análisis de confianza
    
    Returns:
        DataFrame con predicciones
    """
    print("=" * 80)
    print("🎯 PREDICCIÓN DE RETRASOS EN ENTREGAS - OLIST")
    print("=" * 80)
    
    # 1. Cargar modelo
    model_package = load_model(model_path)
    model = model_package['model']
    features = model_package['features']
    
    # 2. Preparar features
    X = prepare_features(order_data, features)
    
    # 3. Realizar predicciones
    print(f"\n🔮 Generando predicciones...")
    predictions = model.predict(X)
    
    # 4. Agregar predicciones al DataFrame original
    result_df = order_data.copy()
    result_df['predicted_delay_days'] = predictions
    
    # 5. Categorizar predicciones
    result_df['delay_category'] = pd.cut(
        predictions,
        bins=[-float('inf'), -2, 0, 3, 7, float('inf')],
        labels=['Muy Adelantado', 'Adelantado', 'A Tiempo', 'Leve Retraso', 'Retraso Grave']
    )
    
    # 6. Clasificación de riesgo
    result_df['risk_level'] = pd.cut(
        predictions,
        bins=[-float('inf'), 0, 3, 7, float('inf')],
        labels=['Bajo', 'Medio', 'Alto', 'Crítico']
    )
    
    # 7. Estadísticas de predicciones
    print(f"\n📊 RESULTADOS:")
    print(f"   • Total predicciones: {len(predictions):,}")
    print(f"   • Delay promedio: {predictions.mean():.2f} días")
    print(f"   • Delay mediano: {np.median(predictions):.2f} días")
    print(f"   • Rango: [{predictions.min():.2f}, {predictions.max():.2f}] días")
    
   
    
    print("=" * 80)
    
    return result_df


def predict_from_csv(
    csv_path: str,
    output_path: str = None,
    model_path: str = None
) -> pd.DataFrame:
    """
    Realiza predicciones desde un archivo CSV.
    
    Args:
        csv_path: Ruta del archivo CSV con datos
        output_path: Ruta para guardar resultados (opcional)
        model_path: Ruta del modelo (si None, usa el más reciente)
    
    Returns:
        DataFrame con predicciones
    """
    # Cargar datos
    print(f"📥 Cargando datos desde: {csv_path}")
    data = pd.read_csv(csv_path)
    print(f"   ✅ {len(data):,} registros cargados")
    
    # Realizar predicciones
    results = predict_delivery_delay(data, model_path)
    
    # Guardar resultados si se especifica
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\n💾 Resultados guardados en: {output_path}")
    
    return results


def get_high_risk_orders(
    predictions_df: pd.DataFrame,
    risk_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Filtra pedidos con alto riesgo de retraso.
    
    Args:
        predictions_df: DataFrame con predicciones
        risk_threshold: Umbral de días para considerar alto riesgo
    
    Returns:
        DataFrame con pedidos de alto riesgo
    """
    high_risk = predictions_df[predictions_df['predicted_delay_days'] >= risk_threshold].copy()
    high_risk = high_risk.sort_values('predicted_delay_days', ascending=False)

    
    if len(high_risk) > 0:
        print(f"   • Delay promedio: {high_risk['predicted_delay_days'].mean():.2f} días")
        print(f"   • Delay máximo: {high_risk['predicted_delay_days'].max():.2f} días")
    
    return high_risk


# ============================================================
# EJEMPLO DE USO
# ============================================================
if __name__ == "__main__":
    """
    Ejemplo de uso del script de predicción.
    """
    
    # Opción 1: Predecir desde CSV de últimos 3 meses
    try:
        print("\n" + "=" * 80)
        print("=" * 80)
        
        # Ruta al CSV de backtesting
        csv_path = 'data/olist_orders_last_3_months.csv'
        
        # Realizar predicciones
        results = predict_from_csv(
            csv_path=csv_path,
            output_path='predictions_output.csv'
        )
        
        # Mostrar muestra de resultados
        display_cols = ['order_id', 'predicted_delay_days', 'delay_category', 'risk_level']
        available_cols = [col for col in display_cols if col in results.columns]
        
        # Identificar pedidos de alto riesgo
        high_risk_orders = get_high_risk_orders(results, risk_threshold=5.0)
        
        if len(high_risk_orders) > 0:
            print("\n🚨 TOP 5 PEDIDOS CON MAYOR RIESGO:")
            risk_display_cols = ['predicted_delay_days', 'delay_category']
            if 'order_id' in high_risk_orders.columns:
                risk_display_cols.insert(0, 'order_id')
            print(high_risk_orders[risk_display_cols].head())
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Asegúrate de que el archivo CSV existe en la ruta especificada.")
    
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ PROCESO COMPLETADO")
    print("=" * 80)