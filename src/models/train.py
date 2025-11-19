# src/models/train.py
"""Módulo para entrenamiento de modelos de clasificación y regresión."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, classification_report, 
    confusion_matrix, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)


def create_preprocessor(X):
    """Crea el preprocessor con imputación y escalado.
    
    Args:
        X: DataFrame con features
        
    Returns:
        ColumnTransformer configurado
    """
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    
    return preprocessor


def train_classification_model(X, y, test_size=0.2, random_state=42):
    """Entrena modelo de clasificación para is_late_delivery.
    
    Args:
        X: Features
        y: Target (is_late_delivery)
        test_size: Proporción de test
        random_state: Semilla aleatoria
        
    Returns:
        dict con modelo, métricas y predicciones
    """
    print("\n" + "="*60)
    print("ENTRENAMIENTO - MODELO DE CLASIFICACIÓN")
    print("="*60)
    
    # Calcular desbalanceo de clases
    class_counts = y.value_counts()
    print(f"\nDistribución de clases:")
    print(f"  Clase 0 (a tiempo): {class_counts[0]} ({class_counts[0]/len(y)*100:.2f}%)")
    print(f"  Clase 1 (retrasado): {class_counts[1]} ({class_counts[1]/len(y)*100:.2f}%)")
    
    scale_pos_weight = class_counts[0] / class_counts[1]
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Crear preprocessor
    preprocessor = create_preprocessor(X)
    
    # Pipeline de clasificación
    model = Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=random_state
            ))
        ]
    )
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    print(f"\nEntrenando XGBClassifier...")
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    results = {
        'model': model,
        'metrics': metrics,
        'y_test': y_test,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return results


def train_regression_models(X, y, test_size=0.2, random_state=42):
    """Entrena y compara múltiples modelos de regresión para delay_days.
    
    Args:
        X: Features
        y: Target (delay_days)
        test_size: Proporción de test
        random_state: Semilla aleatoria
        
    Returns:
        dict con resultados de todos los modelos
    """
    print("\n" + "="*60)
    print("COMPARATIVA DE MODELOS DE REGRESIÓN - delay_days")
    print("="*60)
    
    # Filtrar solo filas válidas
    mask_valid = y.notna()
    X_reg = X[mask_valid]
    y_reg = y[mask_valid]
    
    print(f"\nFilas válidas para regresión: {len(y_reg)} de {len(y)}")
    print(f"Rango de delay_days: [{y_reg.min():.2f}, {y_reg.max():.2f}]")
    print(f"Media de delay_days: {y_reg.mean():.2f}")
    print(f"Mediana de delay_days: {y_reg.median():.2f}")
    
    # Crear preprocessor
    preprocessor = create_preprocessor(X_reg)
    
    # Split train/test
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=test_size, random_state=random_state
    )
    
    # Definir modelos a comparar
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=random_state),
        'Lasso Regression': Lasso(alpha=1.0, random_state=random_state),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            objective='reg:squarederror'
        )
    }
    
    # Entrenar y evaluar modelos
    results = []
    predictions = {}
    
    print("\n" + "="*60)
    print("ENTRENANDO Y EVALUANDO MODELOS...")
    print("="*60)
    
    for model_name, model in models.items():
        print(f"\nEntrenando {model_name}...")
        
        # Crear pipeline
        pipeline = Pipeline([
            ('preproc', preprocessor),
            ('reg', model)
        ])
        
        # Entrenar
        pipeline.fit(X_train_reg, y_train_reg)
        
        # Predecir
        y_pred = pipeline.predict(X_test_reg)
        predictions[model_name] = y_pred
        
        # Calcular métricas
        mse = mean_squared_error(y_test_reg, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_reg, y_pred)
        r2 = r2_score(y_test_reg, y_pred)
        
        results.append({
            'Modelo': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MSE': mse
        })
        
        print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')
    
    # Identificar mejor modelo
    best_model = results_df.iloc[0]['Modelo']
    best_metrics = {
        'model_name': best_model,
        'rmse': results_df.iloc[0]['RMSE'],
        'mae': results_df.iloc[0]['MAE'],
        'r2': results_df.iloc[0]['R²']
    }
    
    return {
        'results_df': results_df,
        'predictions': predictions,
        'best_metrics': best_metrics,
        'y_test': y_test_reg,
        'y_pred_best': predictions[best_model]
    }