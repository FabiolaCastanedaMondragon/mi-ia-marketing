import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="IA Marketing", page_icon="📊")
st.title("📊 Clasificador de Clientes")

@st.cache_resource
def load_resources():
    # Usando tus nombres de archivos confirmados en la captura
    modelo = joblib.load('kmeans_model.pkl') 
    scaler = joblib.load('scaler.pkl')
    return modelo, scaler

try:
    kmeans, scaler = load_resources()
    # TRUCO: Sacar los nombres exactos que el scaler espera
    columnas_entrenamiento = scaler.feature_names_in_
except Exception as e:
    st.error(f"Error al cargar recursos: {e}")

with st.form("main_form"):
    c1, c2 = st.columns(2)
    with c1:
        sales = st.number_input("Ventas ($)", value=1500.0)
        qty = st.number_input("Cantidad", value=20)
        days = st.number_input("Días sin comprar", value=30)
    with c2:
        pais = st.selectbox("País", ["USA", "France", "Spain", "UK", "Australia"])
        cat = st.selectbox("Categoría", ["Classic Cars", "Motorcycles", "Planes", "Ships", "Trains", "Trucks and Buses", "Vintage Cars"])
        size = st.radio("Tamaño", ["Small", "Medium", "Large"])
    
    submit = st.form_submit_button("¡CALCULAR SEGMENTO!")

if submit:
    # Creamos el DataFrame con los nombres EXACTOS que el scaler ya conoce
    df = pd.DataFrame(np.zeros((1, len(columnas_entrenamiento))), columns=columnas_entrenamiento)
    
    # Mapeo manual para asegurar que los datos caigan en la columna correcta
    # (Buscamos el nombre de la columna sin importar si tiene espacios o guiones)
    for col in columnas_entrenamiento:
        if "SALES" in col: df.at[0, col] = sales
        if "QUANTITY" in col: df.at[0, col] = qty
        if "DAYS" in col and "LAST" in col: df.at[0, col] = days
        if "YEAR" in col: df.at[0, col] = 2026
        if "PRICE" in col: df.at[0, col] = 100
        if "MSRP" in col: df.at[0, col] = 100
        
        # Activar categorías (One-Hot Encoding automático)
        if pais == col or cat == col or size == col:
            df.at[0, col] = 1

    try:
        # Predicción
        X_scaled = scaler.transform(df)
        pred = kmeans.predict(X_scaled)[0]
        
        nombres = {0: "💎 VIP", 1: "📦 OCASIONAL", 2: "⭐ FIEL", 3: "⚠️ RIESGO"}
        st.success(f"### RESULTADO: {nombres.get(pred, 'Segmento ' + str(pred))}")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
