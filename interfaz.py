import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="IA Marketing", page_icon="📊")
st.title("📊 Clasificador de Clientes")

# CARGA DE ARCHIVOS CON TUS NOMBRES EXACTOS
@st.cache_resource
def load_resources():
    # Cambié el nombre a kmeans_model.pkl para que coincida con tu captura
    modelo = joblib.load('kmeans_model.pkl') 
    scaler = joblib.load('scaler.pkl')
    return modelo, scaler

try:
    kmeans, scaler = load_resources()
except Exception as e:
    st.error(f"Error al cargar archivos: {e}. Asegúrate de que los nombres en GitHub sean exactos.")

with st.form("main_form"):
    c1, c2 = st.columns(2)
    with c1:
        sales = st.number_input("Ventas ($)", value=1500.0)
        qty = st.number_input("Cantidad", value=20)
        days = st.number_input("Días sin comprar", value=30)
    with c2:
        pais = st.selectbox("País", ["USA", "France", "Spain", "UK", "Australia"])
        cat = st.selectbox("Categoría", ["Classic_Cars", "Motorcycles", "Planes", "Ships"])
        size = st.radio("Tamaño", ["Small", "Medium", "Large"])
    
    submit = st.form_submit_button("¡CALCULAR SEGMENTO!")

if submit:
    # Definimos las 37 columnas que espera tu modelo
    cols = ['QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'SALES', 'MONTH_ID', 'YEAR_ID', 
            'PRODUCTCODE', 'DAYS_SINCE_LASTORDER', 'Australia', 'Austria', 'Belgium', 
            'Canada', 'Denmark', 'Finland', 'France', 'Germany', 'Ireland', 'Italy', 
            'Japan', 'Norway', 'Philippines', 'Singapore', 'Spain', 'Sweden', 
            'Switzerland', 'UK', 'USA', 'Classic_Cars', 'Motorcycles', 'Planes', 
            'Ships', 'Trains', 'Trucks_and_Buses', 'Vintage_Cars', 'Large', 'Medium', 'Small']
    
    # Creamos una fila de ceros
    df = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
    
    # Llenamos los datos básicos del formulario
    df.at[0, 'QUANTITYORDERED'] = qty
    df.at[0, 'SALES'] = sales
    df.at[0, 'DAYS_SINCE_LASTORDER'] = days
    df.at[0, 'PRICEEACH'] = 100 
    df.at[0, 'MSRP'] = 100
    df.at[0, 'YEAR_ID'] = 2026
    
    # Activamos las columnas de categorías (One-Hot Encoding)
    if pais in cols: df.at[0, pais] = 1
    if cat in cols: df.at[0, cat] = 1
    if size in cols: df.at[0, size] = 1

    # Predicción usando el modelo cargado localmente
    X_scaled = scaler.transform(df)
    pred = kmeans.predict(X_scaled)[0]
    
    # Etiquetas de resultado
    nombres = {0: "💎 VIP", 1: "📦 OCASIONAL", 2: "⭐ FIEL", 3: "⚠️ RIESGO"}
    st.success(f"### RESULTADO: {nombres.get(pred, 'Desconocido')}")
    st.info(f"Análisis completado para el cliente de {pais}")
