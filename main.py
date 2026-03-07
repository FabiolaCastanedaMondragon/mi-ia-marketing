from fastapi import FastAPI
import joblib
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel

app = FastAPI()

# Cargamos tus archivos reales
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')
autoencoder = tf.keras.models.load_model('modelo_ia.h5', compile=False)

# Definimos la estructura basada en TU notebook (Ventas de Modelos)
class DatosVenta(BaseModel):
    QUANTITYORDERED: float
    PRICEEACH: float
    SALES: float
    DAYS_SINCE_LASTORDER: float
    # Aquí van las columnas que el scaler espera (los países y categorías)
    # Para la prueba en Swagger, el microservicio usará estos nombres:
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
    Classic_Cars: float = 0
    Motorcycles: float = 0
    Planes: float = 0
    Ships: float = 0
    Trains: float = 0
    Trucks_and_Buses: float = 0
    Vintage_Cars: float = 0

@app.get("/")
def inicio():
    return {"mensaje": "Microservicio de Segmentación de Ventas (Modelos a Escala) Activo"}

@app.post("/predecir")
def predecir(datos: DatosVenta):
    try:
        # Convertir datos a DataFrame
        df = pd.DataFrame([datos.dict()])
        
        # Corregir nombres de columnas para que coincidan con el Scaler (cambiar '_' por ' ')
        df.columns = [c.replace('_', ' ') if '_' in c else c for c in df.columns]
        
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
