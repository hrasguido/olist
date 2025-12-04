import streamlit as st
import json
import os
import glob
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Comparativa ETL Olist", page_icon="📊", layout="wide")
st.title("📊 Comparativa de Métricas de Modelos ETL Olist")
st.markdown("Métricas (MAE, RMSE, R²) por timestamp de ejecución ETL con Prefect (extrae Olist, transforma features, carga Postgres, entrena XGBoost).")

METRICS_DIR = "/app/outputs/model_metrics"

@st.cache_data(ttl=30)
def load_metrics_files():
    pattern = os.path.join(METRICS_DIR, "model_metrics_*.json")
    files = glob.glob(pattern)
    if not files:
        st.warning("No se encontraron archivos con pattern 'model_metrics_*.json'. Ejecuta ETL para generar.")
        return pd.DataFrame()
    
    data = []
    valid_count = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        timestamp_str = filename.replace("model_metrics_", "").replace(".json", "")
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
        except json.JSONDecodeError:
            st.warning(f"JSON inválido en {filename}. Saltando.")
            continue
        
        # Iterar modelos con chequeo robusto
        if isinstance(metrics, dict):
            for model_name, model_data in metrics.items():
                if isinstance(model_data, dict):  # ¡Fix clave! Skip si string
                    row = {
                        'timestamp': timestamp,
                        'model_name': model_name,
                        'mae_train': model_data.get('train', {}).get('mae', None),
                        'rmse_train': model_data.get('train', {}).get('rmse', None),
                        'r2_train': model_data.get('train', {}).get('r2', None),
                        'mae_test': model_data.get('test', {}).get('mae', None),
                        'rmse_test': model_data.get('test', {}).get('rmse', None),
                        'r2_test': model_data.get('test', {}).get('r2', None),
                        'filename': filename
                    }
                    data.append(row)
                    valid_count += 1
                else:
                    st.warning(f"model_data no es dict en {filename} ({model_name}). Saltando modelo.")
        else:
            st.warning(f"Métricas no es dict en {filename}. Saltando archivo.")
    
    st.info(f"Cargados {valid_count} modelos válidos de {len(files)} archivos.")
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('timestamp')
    return df

# Carga y chequeos
df_metrics = load_metrics_files()
if df_metrics.empty:
    st.stop()

# Sidebar filtros
st.sidebar.header("Filtros")
selected_models = st.sidebar.multiselect(
    "Modelos:", options=df_metrics['model_name'].unique(), default=df_metrics['model_name'].unique()
)
min_date, max_date = st.sidebar.date_input(
    "Rango fechas:", value=(df_metrics['timestamp'].min().date(), df_metrics['timestamp'].max().date())
)
df_filtered = df_metrics[
    (df_metrics['model_name'].isin(selected_models)) &
    (df_metrics['timestamp'].dt.date >= min_date) &
    (df_metrics['timestamp'].dt.date <= max_date)
].copy()

if df_filtered.empty:
    st.warning("No datos para filtros.")
    st.stop()

# Tabla comparativa
st.header("📋 Tabla por Timestamp")
pivot_df = df_filtered.pivot_table(
    index=['timestamp', 'model_name'], values=['mae_test', 'rmse_test', 'r2_test'], aggfunc='first'
).reset_index()
pivot_df['timestamp_str'] = pivot_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
st.dataframe(
    pivot_df.style.format({'mae_test': '{:.4f}', 'rmse_test': '{:.4f}', 'r2_test': '{:.4f}'}).background_gradient(cmap='viridis'),
    use_container_width=True
)

# Evolución temporal
st.header("📈 Evolución por Tiempo")
metric = st.selectbox("Métrica:", ['mae_test', 'rmse_test', 'r2_test'])
fig = px.line(
    df_filtered, x='timestamp', y=metric, color='model_name', markers=True,
    title=f"Evolución de {metric.upper()} (desde {df_filtered['timestamp'].min().date()})"
)
st.plotly_chart(fig, use_container_width=True)

# Barras última
st.subheader("🏆 Última Ejecución")
latest_df = df_filtered.loc[df_filtered.groupby('model_name')['timestamp'].idxmax()]
fig_bar = go.Figure()
colors = px.colors.qualitative.Set3
for i, model in enumerate(selected_models):
    model_data = latest_df[latest_df['model_name'] == model]
    if not model_data.empty:
        fig_bar.add_trace(go.Bar(
            name=model, x=['MAE Test', 'RMSE Test', 'R² Test'],
            y=[model_data['mae_test'].iloc[0], model_data['rmse_test'].iloc[0], model_data['r2_test'].iloc[0]],
            marker_color=colors[i % len(colors)]
        ))
fig_bar.update_layout(barmode='group', title="Métricas Test Últimas")
st.plotly_chart(fig_bar, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"*ETL con Prefect (flows/etl_flow.py), Postgres Docker (tablas olist_orders/items). Runs válidos: {len(df_filtered)}.*")
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go

# # Datos de ejemplo: Simula tabla gold_fact_sales de Olist (ventas agregadas)
# data_example = {
#     'order_id': ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010'],
#     'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005', 'CUST006', 'CUST007', 'CUST008', 'CUST009', 'CUST010'],
#     'order_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12', 
#                                  '2023-06-18', '2023-07-22', '2023-08-30', '2023-09-14', '2023-10-28']),
#     'customer_state': ['SP', 'RJ', 'MG', 'SP', 'RJ', 'SP', 'MG', 'SP', 'RJ', 'MG'],
#     'total_amount': [150.50, 250.00, 180.75, 300.25, 120.00, 220.50, 190.00, 280.75, 160.25, 210.00],
#     'payment_type': ['credit_card', 'boleto', 'credit_card', 'debit_card', 'boleto', 'credit_card', 'debit_card', 'credit_card', 'boleto', 'debit_card']
# }

# df = pd.DataFrame(data_example)

# # Configuración de la página
# st.set_page_config(page_title="Dashboard ETL Olist - Ejemplo", layout="wide")
# st.title("🚀 Dashboard de Ventas Olist (Datos de Ejemplo)")
# st.markdown("Este es un ejemplo independiente: visualiza datos simulados de ventas sin conexión a Prefect o Postgres. ¡Prueba filtros y gráficos!")

# # Sidebar para filtros
# st.sidebar.header("Filtros")
# selected_states = st.sidebar.multiselect("Estado del Cliente:", options=df['customer_state'].unique(), default=df['customer_state'].unique())
# filtered_df = df[df['customer_state'].isin(selected_states)]

# # Métricas clave
# col1, col2, col3 = st.columns(3)
# col1.metric("Total de Órdenes", len(filtered_df))
# col2.metric("Monto Total de Ventas", f"R$ {filtered_df['total_amount'].sum():.2f}")
# col3.metric("Órdenes Promedio por Cliente", f"{len(filtered_df) / len(filtered_df['customer_id'].unique()):.1f}")

# # Tabla de datos
# st.subheader("📊 Tabla de Ventas")
# st.dataframe(filtered_df, use_container_width=True)

# # Gráficos
# col_left, col_right = st.columns(2)

# with col_left:
#     st.subheader("📈 Ventas por Estado (Barras)")
#     state_sales = filtered_df.groupby('customer_state')['total_amount'].sum().reset_index()
#     fig_bar = px.bar(state_sales, x='customer_state', y='total_amount', title="Total Ventas por Estado",
#                      color='total_amount', color_continuous_scale='Viridis')
#     st.plotly_chart(fig_bar, use_container_width=True)

# with col_right:
#     st.subheader("📉 Tendencia de Ventas por Fecha (Líneas)")
#     monthly_sales = filtered_df.resample('M', on='order_date')['total_amount'].sum().reset_index()
#     fig_line = px.line(monthly_sales, x='order_date', y='total_amount', title="Ventas Mensuales",
#                        markers=True)
#     st.plotly_chart(fig_line, use_container_width=True)

# # Pie chart extra para tipos de pago
# st.subheader("💳 Distribución por Tipo de Pago")
# payment_dist = filtered_df['payment_type'].value_counts()
# fig_pie = px.pie(values=payment_dist.values, names=payment_dist.index, title="Tipos de Pago")
# st.plotly_chart(fig_pie, use_container_width=True)

# st.markdown("---")
# st.caption("💡 *Ejemplo estático para pruebas. Para datos reales, integra con Postgres en `conn.py` y reemplaza `data_example` por una query.*")