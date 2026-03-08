import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="IA Marketing - Fabiola", page_icon="📊")

# Título visual
st.title("📊 Clasificador de Clientes Inteligente")
st.markdown("### Ingeniería de Sistemas - Segmentación RFM")

# Carga de recursos con caché para que sea rápido
@st.cache_resource
def load_resources():
    modelo = joblib.load('kmeans_model.pkl') 
    scaler = joblib.load('scaler.pkl')
    return modelo, scaler

try:
    kmeans, scaler = load_resources()
    columnas_entrenamiento = scaler.feature_names_in_ # Mapeo automático de columnas
except Exception as e:
    st.error(f"Error al cargar los archivos .pkl: {e}")

# Formulario de entrada de datos
with st.form("main_form"):
    st.subheader("📝 Datos del Cliente")
    c1, c2 = st.columns(2)
    
    with c1:
        sales = st.number_input("Ventas Totales ($)", value=1500.0, help="Monto total vendido al cliente")
        qty = st.number_input("Cantidad de Productos", value=20)
        days = st.number_input("Días desde la última compra", value=30)
    
    with c2:
        pais = st.selectbox("País de Origen", ["USA", "France", "Spain", "UK", "Australia", "Norway", "Germany"])
        cat = st.selectbox("Categoría de Producto", ["Classic Cars", "Motorcycles", "Planes", "Ships", "Trains", "Trucks and Buses", "Vintage Cars"])
        size = st.radio("Tamaño del Pedido", ["Small", "Medium", "Large"])
    
    submit = st.form_submit_button("🚀 ¡CALCULAR SEGMENTO!")

if submit:
    # Crear DataFrame base con ceros
    df = pd.DataFrame(np.zeros((1, len(columnas_entrenamiento))), columns=columnas_entrenamiento)
    
    # Llenado dinámico según las columnas que espera el modelo
    for col in columnas_entrenamiento:
        if "SALES" in col: df.at[0, col] = sales
        if "QUANTITY" in col: df.at[0, col] = qty
        if "DAYS" in col and "LAST" in col: df.at[0, col] = days
        if "YEAR" in col: df.at[0, col] = 2026
        if "PRICE" in col: df.at[0, col] = 100
        if "MSRP" in col: df.at[0, col] = 100
        # Mapeo de variables categóricas (One-Hot Encoding manual)
        if pais == col or cat == col or size == col:
            df.at[0, col] = 1

    try:
        # Proceso de Predicción
        X_scaled = scaler.transform(df)
        cluster_id = kmeans.predict(X_scaled)[0]
        
        # DICCIONARIO CORREGIDO SEGÚN TUS PRUEBAS REALES
        # Cluster 2 = Rico ($900k) -> VIP
        # Cluster 1 = Pobre ($1, 3000 días) -> RIESGO
        nombres_segmentos = {
            2: "💎 CLIENTE VIP",
            1: "⚠️ CLIENTE EN RIESGO",
            0: "⭐ CLIENTE FIEL",
            3: "📦 CLIENTE OCASIONAL"
        }
        
        resultado_final = nombres_segmentos.get(cluster_id, f"Segmento {cluster_id}")
        
        # Mostrar resultado con estilo
        st.markdown("---")
        st.success(f"## RESULTADO: {resultado_final}")
        st.info(f"Análisis técnico: El algoritmo K-Means asignó el Cluster ID: {cluster_id}")
        
    except Exception as e:
        st.error(f"Error técnico en la predicción: {e}")

# Pie de página técnico
st.caption("Desarrollado para el proyecto Wise Agend - Módulo de Inteligencia de Marketing.")
