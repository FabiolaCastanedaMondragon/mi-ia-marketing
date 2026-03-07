from fastapi import FastAPI
import joblib
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel

app = FastAPI()

# Cargamos tus archivos
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')
autoencoder = tf.keras.models.load_model('modelo_ia.h5', compile=False)

# Definimos la estructura exacta con TODAS las piezas que pide tu modelo
class DatosVenta(BaseModel):
    ORDERLINENUMBER: float = 0
    QUANTITYORDERED: float = 0
    PRICEEACH: float = 0
    MSRP: float = 0
    SALES: float = 0
    MONTH_ID: float = 0
    DAYS_SINCE_LASTORDER: float = 0
    # Países
    Australia: float = 0
    Austria: float = 0
    Belgium: float = 0
    Canada: float = 0
    Denmark: float = 0
    Finland: float = 0
    France: float = 0
    Germany: float = 0
    Ireland: float = 0
    Italy: float = 0
    Japan: float = 0
    Norway: float = 0
    Philippines: float = 0
    Singapore: float = 0
    Spain: float = 0
    Sweden: float = 0
    Switzerland: float = 0
    UK: float = 0
    USA: float = 0
    # Categorías
    Classic_Cars: float = 0
    Motorcycles: float = 0
    Planes: float = 0
    Ships: float = 0
    Trains: float = 0
    Trucks_and_Buses: float = 0
    Vintage_Cars: float = 0
    # Tamaños (Estas son las que faltaban)
    Large: float = 0
    Medium: float = 0
    Small: float = 0

@app.get("/")
def inicio():
    return {"mensaje": "Microservicio de Segmentación de Ventas Activo"}

@app.post("/predecir")
def predecir(datos: DatosVenta):
    try:
        # Convertir a diccionario
        data_dict = datos.dict()
        
        # Crear DataFrame
        df = pd.DataFrame([data_dict])
        
        # REGLA DE ORO: Los nombres deben ser EXACTOS a como se entrenaron
        # Cambiamos guiones bajos por espacios donde el modelo lo necesite
        mapeo_nombres = {
            'Classic_Cars': 'Classic Cars',
            'Trucks_and_Buses': 'Trucks and Buses',
            'Vintage_Cars': 'Vintage Cars',
            'DAYS_SINCE_LASTORDER': 'DAYS SINCE LASTORDER'
        }
        df = df.rename(columns=mapeo_nombres)
        
        # 1. Escalado
        datos_escalados = scaler.transform(df)
        
        # 2. Predicción de Cluster
        grupo = kmeans.predict(datos_escalados)
        
        return {
            "segmento_venta": int(grupo[0]),
            "status": "Exitoso"
        }
    except Exception as e:
        return {"error": str(e)}
