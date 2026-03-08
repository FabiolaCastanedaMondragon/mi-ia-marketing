import streamlit as st
import pandas as pd
import joblib

# Configuración
st.set_page_config(page_title="IA de Marketing", page_icon="📊")
st.title("📊 Clasificador Inteligente de Clientes")

# CARGAR EL MODELO DIRECTAMENTE (Asegúrate que los nombres coincidan con tus archivos)
@st.cache_resource
def load_models():
    modelo = joblib.load('modelo_kmeans.pkl')
    scaler = joblib.load('scaler.pkl')
    return modelo, scaler

try:
    kmeans, scaler = load_models()
except:
    st.error("No se encontraron los archivos .pkl en GitHub")

with st.form("formulario"):
    col1, col2 = st.columns(2)
    with col1:
        sales = st.number_input("Monto de la Venta ($)", value=1000.0)
        quantity = st.number_input("Cantidad", value=10)
        days = st.number_input("Días desde última compra", value=30)
    with col2:
        pais = st.selectbox("País", ["USA", "France", "Spain", "UK", "Australia"])
        cat = st.selectbox("Categoría", ["Classic_Cars", "Motorcycles", "Planes", "Ships"])
        size = st.radio("Tamaño", ["Small", "Medium", "Large"])
    
    btn = st.form_submit_button("Analizar Cliente")

if btn:
    # Crear el DataFrame con las 37 columnas que pide tu modelo
    # (Usa la misma lógica de ceros y unos que tenías en main.py)
    input_data = pd.DataFrame([[quantity, 100, 100, sales, 1, 2026, 1, days] + [0]*29], 
                              columns=['QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'SALES', 'MONTH_ID', 'YEAR_ID', 'PRODUCTCODE', 'DAYS_SINCE_LASTORDER', 'Australia', 'Austria', 'Belgium', 'Canada', 'Denmark', 'Finland', 'France', 'Germany', 'Ireland', 'Italy', 'Japan', 'Norway', 'Philippines', 'Singapore', 'Spain', 'Sweden', 'Switzerland', 'UK', 'USA', 'Classic_Cars', 'Motorcycles', 'Planes', 'Ships', 'Trains', 'Trucks_and_Buses', 'Vintage_Cars', 'Large', 'Medium', 'Small'])
    
    # Lógica de asignación de variables (USA, France, etc.)
    if pais in input_data.columns: input_data[pais] = 1
    if cat in input_data.columns: input_data[cat] = 1
    if size in input_data.columns: input_data[size] = 1

    # Predicción
    X_scaled = scaler.transform(input_data)
    segmento = kmeans.predict(X_scaled)[0]
    
    nombres = {0: "💎 VIP", 1: "📦 OCASIONAL", 2: "⭐ FIEL", 3: "⚠️ EN RIESGO"}
    st.success(f"### Resultado: {nombres.get(segmento)}")
