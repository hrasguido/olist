# src/models/evaluate.py
"""Módulo para evaluación e impresión de métricas."""


def print_classification_metrics(results):
    """Imprime métricas de clasificación.
    
    Args:
        results: dict con métricas del modelo de clasificación
    """
    metrics = results['metrics']
    
    print("\n" + "="*50)
    print("MÉTRICAS DE CLASIFICACIÓN (is_late_delivery)")
    print("="*50)
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"ROC AUC:    {metrics['roc_auc']:.4f}  <- Métrica principal para desbalanceo")
    print(f"F1 Score:   {metrics['f1']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}  (de los predichos como retrasados, cuántos lo son)")
    print(f"Recall:     {metrics['recall']:.4f}  (de los realmente retrasados, cuántos detectamos)")
    print("="*50)
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])


def print_regression_metrics(results):
    """Imprime comparativa de métricas de regresión.
    
    Args:
        results: dict con resultados de modelos de regresión
    """
    results_df = results['results_df']
    best_metrics = results['best_metrics']
    
    print("\n" + "="*60)
    print("COMPARATIVA DE MÉTRICAS DE REGRESIÓN")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    print(f"\n🏆 MEJOR MODELO: {best_metrics['model_name']}")
    print(f"   RMSE: {best_metrics['rmse']:.4f} días")
    print(f"   MAE:  {best_metrics['mae']:.4f} días")
    print(f"   R²:   {best_metrics['r2']:.4f}")


def print_residuals_analysis(y_test, y_pred, model_name):
    """Imprime análisis de residuos.
    
    Args:
        y_test: Valores reales
        y_pred: Predicciones
        model_name: Nombre del modelo
    """
    residuals = y_test - y_pred
    
    print("\n" + "="*50)
    print(f"ANÁLISIS DE RESIDUOS ({model_name})")
    print("="*50)
    print(f"Media de residuos:      {residuals.mean():.4f}")
    print(f"Std de residuos:        {residuals.std():.4f}")
    print(f"Residuo mínimo:         {residuals.min():.4f}")
    print(f"Residuo máximo:         {residuals.max():.4f}")
    print("="*50)