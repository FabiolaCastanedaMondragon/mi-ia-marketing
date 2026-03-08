import streamlit as st
import requests

# Configuración de la página
st.set_page_config(page_title="IA de Marketing - Fabiola", page_icon="📊")

st.title("📊 Clasificador Inteligente de Clientes")
st.markdown("Introduce los datos de la venta para que la IA determine el segmento del cliente.")

# Formulario organizado
with st.form("formulario_prediccion"):
    col1, col2 = st.columns(2)
    
    with col1:
        sales = st.number_input("Monto de la Venta ($)", min_value=0.0, value=1000.0)
        quantity = st.number_input("Cantidad de productos", min_value=1, value=10)
        days = st.number_input("Días desde la última compra", min_value=0, value=30)
    
    with col2:
        pais = st.selectbox("País del cliente", ["USA", "France", "Spain", "UK", "Australia", "Others"])
        categoria = st.selectbox("Categoría de Producto", ["Classic_Cars", "Motorcycles", "Planes", "Ships", "Trains"])
        size = st.radio("Tamaño del pedido", ["Small", "Medium", "Large"])

    btn_predecir = st.form_submit_button("Analizar Cliente")

# Lógica al presionar el botón
if btn_predecir:
    # Preparamos los datos (poniendo 1 en el seleccionado y 0 en los demás)
    datos = {
        "QUANTITYORDERED": quantity, "PRICEEACH": 100, "MSRP": 100, "SALES": sales,
        "MONTH_ID": 1, "YEAR_ID": 2026, "PRODUCTCODE": 1, "DAYS_SINCE_LASTORDER": days,
        "Australia": 1 if pais == "Australia" else 0, "Austria": 0, "Belgium": 0, "Canada": 0, "Denmark": 0, "Finland": 0,
        "France": 1 if pais == "France" else 0, "Germany": 0, "Ireland": 0, "Italy": 0, "Japan": 0, "Norway": 0,
        "Philippines": 0, "Singapore": 0, "Spain": 1 if pais == "Spain" else 0, "Sweden": 0, "Switzerland": 0,
        "UK": 1 if pais == "UK" else 0, "USA": 1 if pais == "USA" else 0,
        "Classic_Cars": 1 if categoria == "Classic_Cars" else 0, "Motorcycles": 1 if categoria == "Motorcycles" else 0,
        "Planes": 1 if categoria == "Planes" else 0, "Ships": 1 if categoria == "Ships" else 0,
        "Trains": 1 if categoria == "Trains" else 0, "Trucks_and_Buses": 0, "Vintage_Cars": 0,
        "Large": 1 if size == "Large" else 0, "Medium": 1 if size == "Medium" else 0, "Small": 1 if size == "Small" else 0
    }

    try:
        # CONEXIÓN CON TU API EN RENDER
        url_api = "https://mi-ia-marketing.onrender.com/predecir"
        respuesta = requests.post(url_api, json=datos)
        resultado = respuesta.json()
        
        segmento = resultado["segmento_venta"]
        
        # Traducción de números a palabras bonitas
        nombres = {
            0: "💎 CLIENTE VIP (ALTO VALOR)",
            1: "📦 CLIENTE OCASIONAL (BAJO VALOR)",
            2: "⭐ CLIENTE FIEL (VALOR MEDIO)",
            3: "⚠️ CLIENTE EN RIESGO"
        }

        st.success(f"### Resultado: {nombres.get(segmento, 'Desconocido')}")
        st.info(f"La IA clasificó este perfil en el Grupo {segmento}")
        
    except Exception as e:
        st.error("Error al conectar con la IA. Verifica que el servidor esté prendido.")