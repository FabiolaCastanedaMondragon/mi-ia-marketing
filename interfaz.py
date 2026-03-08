import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="IA Marketing", page_icon="📊")
st.title("📊 Clasificador de Clientes")

@st.cache_resource
def load_resources():
    modelo = joblib.load('kmeans_model.pkl') 
    scaler = joblib.load('scaler.pkl')
    return modelo, scaler

try:
    kmeans, scaler = load_resources()
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
    df = pd.DataFrame(np.zeros((1, len(columnas_entrenamiento))), columns=columnas_entrenamiento)
    
    for col in columnas_entrenamiento:
        if "SALES" in col: df.at[0, col] = sales
        if "QUANTITY" in col: df.at[0, col] = qty
        if "DAYS" in col and "LAST" in col: df.at[0, col] = days
        if "YEAR" in col: df.at[0, col] = 2026
        if "PRICE" in col: df.at[0, col] = 100
        if "MSRP" in col: df.at[0, col] = 100
        if pais == col or cat == col or size == col:
            df.at[0, col] = 1

    try:
        X_scaled = scaler.transform(df)
        pred = kmeans.predict(X_scaled)[0]
        
        # ESTO ES LO IMPORTANTE: Nos dirá el número real
        st.info(f"🔢 El modelo clasificó este cliente en el CLUSTER: {pred}")
        
        # Diccionario temporal para que no se rompa, lo corregiremos con tus resultados
        nombres = {0: "Grupo 0", 1: "Grupo 1", 2: "Grupo 2", 3: "Grupo 3"}
        st.success(f"### RESULTADO ACTUAL: {nombres.get(pred)}")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
