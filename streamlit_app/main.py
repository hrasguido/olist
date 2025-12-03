import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Datos de ejemplo: Simula tabla gold_fact_sales de Olist (ventas agregadas)
data_example = {
    'order_id': ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010'],
    'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005', 'CUST006', 'CUST007', 'CUST008', 'CUST009', 'CUST010'],
    'order_date': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12', 
                                 '2023-06-18', '2023-07-22', '2023-08-30', '2023-09-14', '2023-10-28']),
    'customer_state': ['SP', 'RJ', 'MG', 'SP', 'RJ', 'SP', 'MG', 'SP', 'RJ', 'MG'],
    'total_amount': [150.50, 250.00, 180.75, 300.25, 120.00, 220.50, 190.00, 280.75, 160.25, 210.00],
    'payment_type': ['credit_card', 'boleto', 'credit_card', 'debit_card', 'boleto', 'credit_card', 'debit_card', 'credit_card', 'boleto', 'debit_card']
}

df = pd.DataFrame(data_example)

# Configuración de la página
st.set_page_config(page_title="Dashboard ETL Olist - Ejemplo", layout="wide")
st.title("🚀 Dashboard de Ventas Olist (Datos de Ejemplo)")
st.markdown("Este es un ejemplo independiente: visualiza datos simulados de ventas sin conexión a Prefect o Postgres. ¡Prueba filtros y gráficos!")

# Sidebar para filtros
st.sidebar.header("Filtros")
selected_states = st.sidebar.multiselect("Estado del Cliente:", options=df['customer_state'].unique(), default=df['customer_state'].unique())
filtered_df = df[df['customer_state'].isin(selected_states)]

# Métricas clave
col1, col2, col3 = st.columns(3)
col1.metric("Total de Órdenes", len(filtered_df))
col2.metric("Monto Total de Ventas", f"R$ {filtered_df['total_amount'].sum():.2f}")
col3.metric("Órdenes Promedio por Cliente", f"{len(filtered_df) / len(filtered_df['customer_id'].unique()):.1f}")

# Tabla de datos
st.subheader("📊 Tabla de Ventas")
st.dataframe(filtered_df, use_container_width=True)

# Gráficos
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Ventas por Estado (Barras)")
    state_sales = filtered_df.groupby('customer_state')['total_amount'].sum().reset_index()
    fig_bar = px.bar(state_sales, x='customer_state', y='total_amount', title="Total Ventas por Estado",
                     color='total_amount', color_continuous_scale='Viridis')
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("📉 Tendencia de Ventas por Fecha (Líneas)")
    monthly_sales = filtered_df.resample('M', on='order_date')['total_amount'].sum().reset_index()
    fig_line = px.line(monthly_sales, x='order_date', y='total_amount', title="Ventas Mensuales",
                       markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

# Pie chart extra para tipos de pago
st.subheader("💳 Distribución por Tipo de Pago")
payment_dist = filtered_df['payment_type'].value_counts()
fig_pie = px.pie(values=payment_dist.values, names=payment_dist.index, title="Tipos de Pago")
st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
st.caption("💡 *Ejemplo estático para pruebas. Para datos reales, integra con Postgres en `conn.py` y reemplaza `data_example` por una query.*")