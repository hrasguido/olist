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

def run_pipeline(orders_file=None):
    DATASET_DIR = 'data/raw/'

    datasets = {}

    # 1. Load data
    customers = pd.read_csv(os.path.join(DATASET_DIR, 'olist_customers_dataset.csv'))
    geolocation = pd.read_csv(os.path.join(DATASET_DIR, 'olist_geolocation_dataset.csv'))
    order_items = pd.read_csv(os.path.join(DATASET_DIR, 'olist_order_items_dataset.csv'))
    order_payments = pd.read_csv(os.path.join(DATASET_DIR, 'olist_order_payments_dataset.csv'))
    order_reviews = pd.read_csv(os.path.join(DATASET_DIR, 'olist_order_reviews_dataset.csv'))
    orders = pd.read_csv(os.path.join(DATASET_DIR, orders_file))
    products = pd.read_csv(os.path.join(DATASET_DIR, 'olist_products_dataset.csv'))
    sellers = pd.read_csv(os.path.join(DATASET_DIR, 'olist_sellers_dataset.csv'))
    category_translation = pd.read_csv(os.path.join(DATASET_DIR, 'product_category_name_translation.csv'))

    # 2. Clean data
    datasets['customers'] = customers
    datasets['geolocation'] = geolocation
    datasets['order_items'] = clean_module.clean_order_items(order_items)
    datasets['order_payments'] = clean_module.clean_order_payments(order_payments)
    datasets['order_reviews'] = clean_module.clean_order_reviews(order_reviews)
    datasets['orders'] = clean_module.clean_orders(orders)
    datasets['products'] = clean_module.clean_products(products)
    datasets['sellers'] = sellers
    datasets['category_translation'] = category_translation

    initial_master_df = pd.merge(datasets['orders'], datasets['customers'], on='customer_id', how='left')
    initial_master_df = initial_master_df.merge(datasets['order_items'], on='order_id', how='left')
    initial_master_df = initial_master_df.merge(datasets['sellers'], on='seller_id', how='left')
    initial_master_df = initial_master_df.merge(datasets['products'], on='product_id', how='left')
    initial_master_df = initial_master_df.merge(datasets['category_translation'], on='product_category_name', how='left')
    initial_master_df = initial_master_df.merge(datasets['order_payments'], on='order_id', how='left')
    initial_master_df = initial_master_df.merge(datasets['order_reviews'], on='order_id', how='left')

    # 3. Feature engineering | 58
    features_and_targets_df = features_module.build_features_and_targets(datasets)

    master_df = features_module.create_master_df(initial_master_df, features_and_targets_df)

    # # 4. Split X, y1, y2
    X = master_df.drop(["is_late_delivery", "delay_days"], axis=1)
    y1 = master_df["is_late_delivery"]
    y2 = master_df["delay_days"]

    # Calcular el desbalanceo de clases
    class_counts = y1.value_counts()
    print(f"\nDistribución de clases:")
    print(f"  Clase 0 (a tiempo): {class_counts[0]} ({class_counts[0]/len(y1)*100:.2f}%)")
    print(f"  Clase 1 (retrasado): {class_counts[1]} ({class_counts[1]/len(y1)*100:.2f}%)")
    
    # Calcular scale_pos_weight para XGBoost
    scale_pos_weight = class_counts[0] / class_counts[1]
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # 5. Preprocessing pipeline con imputación de valores faltantes
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

    # 6. Full pipeline = preprocessing + modelo
    model = Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,  # Manejo de desbalanceo
                use_label_encoder=True,
                eval_metric='logloss',
                random_state=42
            ))
        ]
    )

    # 7. Split train/test
    #X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, stratify=y1, random_state=42)


    # 8. Entrenar pipeline completo
    model.fit(X_train, y_train)

    # 9. Evaluación
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MÉTRICAS DE CLASIFICACIÓN (is_late_delivery)")
    print("="*50)
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"ROC AUC:    {roc_auc:.4f}  <- Métrica principal para desbalanceo")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Precision:  {precision:.4f}  (de los predichos como retrasados, cuántos lo son)")
    print(f"Recall:     {recall:.4f}  (de los realmente retrasados, cuántos detectamos)")
    print("="*50)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # ========================================
    # COMPARATIVA DE MODELOS DE REGRESIÓN PARA delay_days
    # ========================================
    print("\n" + "="*60)
    print("COMPARATIVA DE MODELOS DE REGRESIÓN - delay_days")
    print("="*60)
    
    # Filtrar solo las filas donde delay_days no es NaN
    mask_valid = y2.notna()
    X_reg = X[mask_valid]
    y_reg = y2[mask_valid]
    
    print(f"\nFilas válidas para regresión: {len(y_reg)} de {len(y2)}")
    print(f"Rango de delay_days: [{y_reg.min():.2f}, {y_reg.max():.2f}]")
    print(f"Media de delay_days: {y_reg.mean():.2f}")
    print(f"Mediana de delay_days: {y_reg.median():.2f}")
    
    # Split train/test para regresión
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Definir modelos a comparar
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=1.0, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, objective='reg:squarederror')
    }
    
    # Almacenar resultados
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
    
    print("\n" + "="*60)
    print("COMPARATIVA DE MÉTRICAS DE REGRESIÓN")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)
    
    # Identificar mejor modelo
    best_model = results_df.iloc[0]['Modelo']
    best_rmse = results_df.iloc[0]['RMSE']
    best_mae = results_df.iloc[0]['MAE']
    best_r2 = results_df.iloc[0]['R²']
    
    print(f"\n🏆 MEJOR MODELO: {best_model}")
    print(f"   RMSE: {best_rmse:.4f} días")
    print(f"   MAE:  {best_mae:.4f} días")
    print(f"   R²:   {best_r2:.4f}")
    
    # Usar predicciones del mejor modelo para visualización
    y_pred_reg = predictions[best_model]
    rmse = best_rmse
    mae = best_mae
    r2 = best_r2
    
    # Visualización de predicciones vs reales
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.5, s=20)
    plt.plot([y_test_reg.min(), y_test_reg.max()], 
             [y_test_reg.min(), y_test_reg.max()], 
             'r--', lw=2, label='Predicción perfecta')
    plt.xlabel('Delay Days Real', fontsize=12)
    plt.ylabel('Delay Days Predicho', fontsize=12)
    plt.title('Predicciones vs Valores Reales', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Residuos
    plt.subplot(1, 2, 2)
    residuals = y_test_reg - y_pred_reg
    plt.scatter(y_pred_reg, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Delay Days Predicho', fontsize=12)
    plt.ylabel('Residuos', fontsize=12)
    plt.title('Análisis de Residuos', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nGráficos de regresión guardados en: outputs/regression_analysis.png")
    
    # Estadísticas de residuos
    print("\n" + "="*50)
    print("ANÁLISIS DE RESIDUOS")
    print("="*50)
    print(f"Media de residuos:      {residuals.mean():.4f}")
    print(f"Std de residuos:        {residuals.std():.4f}")
    print(f"Residuo mínimo:         {residuals.min():.4f}")
    print(f"Residuo máximo:         {residuals.max():.4f}")
    print("="*50)

    # 10. Guardar pipeline entrenado
    import joblib
    joblib.dump(model, f"models/model_{month}.pkl")
     
run_pipeline('olist_orders_before_3_months.csv')