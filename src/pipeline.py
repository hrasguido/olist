# src/pipeline.py
"""Pipeline refactorizado para entrenamiento de modelos."""

import pandas as pd
import os
import warnings
import importlib.util
import joblib

# Importar módulos locales
from models.train import train_classification_model, train_regression_models
from models.evaluate import print_classification_metrics, print_regression_metrics, print_residuals_analysis
from models.visualize import plot_regression_analysis, plot_model_comparison, plot_predictions_comparison

warnings.filterwarnings("ignore")

# Cargar módulos de limpieza y features
clean_path = os.path.abspath("../code/src/data/clean.py")
spec = importlib.util.spec_from_file_location("clean", clean_path)
clean_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean_module)

features_path = os.path.abspath("../code/src/features/make_features.py")
spec = importlib.util.spec_from_file_location("features", features_path)
features_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(features_module)

def run_pipeline(orders_file='olist_orders_dataset.csv', save_models=True, output_dir='outputs/'):
    """Ejecuta el pipeline completo de ML.
    
    Args:
        orders_file: Nombre del archivo de órdenes
        save_models: Si True, guarda los modelos entrenados
        output_dir: Directorio para outputs
    
    Returns:
        dict con resultados de clasificación y regresión
    """
    print("\n" + "="*70)
    print("  PIPELINE DE MACHINE LEARNING - PREDICCIÓN DE ENTREGAS")
    print("="*70)
    
    # Crear directorios
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    DATASET_DIR = 'data/raw/'
    datasets = {}

    # 1. Cargar datos
    print("\n" + "="*60)
    print("CARGANDO DATOS")
    print("="*60)
    
    datasets['customers'] = pd.read_csv(os.path.join(DATASET_DIR, 'olist_customers_dataset.csv'))
    datasets['geolocation'] = pd.read_csv(os.path.join(DATASET_DIR, 'olist_geolocation_dataset.csv'))
    datasets['order_items'] = pd.read_csv(os.path.join(DATASET_DIR, 'olist_order_items_dataset.csv'))
    datasets['order_payments'] = pd.read_csv(os.path.join(DATASET_DIR, 'olist_order_payments_dataset.csv'))
    datasets['order_reviews'] = pd.read_csv(os.path.join(DATASET_DIR, 'olist_order_reviews_dataset.csv'))
    datasets['orders'] = pd.read_csv(os.path.join(DATASET_DIR, orders_file))
    datasets['products'] = pd.read_csv(os.path.join(DATASET_DIR, 'olist_products_dataset.csv'))
    datasets['sellers'] = pd.read_csv(os.path.join(DATASET_DIR, 'olist_sellers_dataset.csv'))
    datasets['category_translation'] = pd.read_csv(os.path.join(DATASET_DIR, 'product_category_name_translation.csv'))
    
    print(f"✅ {len(datasets)} datasets cargados")

    # 2. Limpiar datos
    print("\n" + "="*60)
    print("LIMPIANDO DATOS")
    print("="*60)
    
    datasets['order_items'] = clean_module.clean_order_items(datasets['order_items'])
    datasets['order_payments'] = clean_module.clean_order_payments(datasets['order_payments'])
    datasets['order_reviews'] = clean_module.clean_order_reviews(datasets['order_reviews'])
    datasets['orders'] = clean_module.clean_orders(datasets['orders'])
    datasets['products'] = clean_module.clean_products(datasets['products'])
    
    print("✅ Datos limpios")

    # 3. Crear master dataframe
    print("\n" + "="*60)
    print("GENERANDO FEATURES")
    print("="*60)
    
    initial_master_df = pd.merge(datasets['orders'], datasets['customers'], on='customer_id', how='left')
    initial_master_df = initial_master_df.merge(datasets['order_items'], on='order_id', how='left')
    initial_master_df = initial_master_df.merge(datasets['sellers'], on='seller_id', how='left')
    initial_master_df = initial_master_df.merge(datasets['products'], on='product_id', how='left')
    initial_master_df = initial_master_df.merge(datasets['category_translation'], on='product_category_name', how='left')
    initial_master_df = initial_master_df.merge(datasets['order_payments'], on='order_id', how='left')
    initial_master_df = initial_master_df.merge(datasets['order_reviews'], on='order_id', how='left')

    features_and_targets_df = features_module.build_features_and_targets(datasets)
    master_df = features_module.create_master_df(initial_master_df, features_and_targets_df)
    
    print(f"✅ Master DataFrame: {master_df.shape}")

    # 4. Preparar datos
    X = master_df.drop(["is_late_delivery", "delay_days"], axis=1)
    y_classification = master_df["is_late_delivery"]
    y_regression = master_df["delay_days"]
    
    # 5. Entrenar modelo de clasificación
    clf_results = train_classification_model(X, y_classification)
    print_classification_metrics(clf_results)
    
    # 6. Entrenar modelos de regresión
    reg_results = train_regression_models(X, y_regression)
    print_regression_metrics(reg_results)
    print_residuals_analysis(reg_results['y_test'], reg_results['y_pred_best'], reg_results['best_metrics']['model_name'])
    
    # 7. Generar visualizaciones
    print("\n" + "="*60)
    print("GENERANDO VISUALIZACIONES")
    print("="*60)
    
    plot_regression_analysis(reg_results, output_dir)
    plot_model_comparison(reg_results, output_dir)
    plot_predictions_comparison(reg_results, output_dir)
    
    # Guardar resultados en CSV
    reg_results['results_df'].to_csv(os.path.join(output_dir, 'regression_results.csv'), index=False)
    print(f"✅ Resultados guardados en: {os.path.join(output_dir, 'regression_results.csv')}")
    
    # 8. Guardar modelos
    if save_models:
        model_prefix = orders_file.replace('.csv', '')
        clf_path = f"models/{model_prefix}_classification.pkl"
        joblib.dump(clf_results['model'], clf_path)
        print(f"✅ Modelo de clasificación guardado: {clf_path}")
    
    # Resumen final
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"\n🏆 Clasificación (is_late_delivery):")
    print(f"   - ROC AUC: {clf_results['metrics']['roc_auc']:.4f}")
    print(f"   - F1 Score: {clf_results['metrics']['f1']:.4f}")
    print(f"\n🏆 Regresión (delay_days): {reg_results['best_metrics']['model_name']}")
    print(f"   - RMSE: {reg_results['best_metrics']['rmse']:.4f} días")
    print(f"   - R²: {reg_results['best_metrics']['r2']:.4f}")
    print("\n" + "="*70)
    
    return {
        'classification': clf_results,
        'regression': reg_results,
        'master_df': master_df
    }


if __name__ == "__main__":
    results = run_pipeline('olist_orders_before_3_months.csv')