from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# --- CONFIGURACIÓN DE SEGURIDAD (CORS) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga de modelos
try:
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    # autoencoder = tf.keras.models.load_model('modelo_ia.h5', compile=False) # Solo si lo usas
except Exception as e:
    print(f"Error cargando modelos: {e}")

# Esquema de datos (37 columnas)
class DatosVenta(BaseModel):
    QUANTITYORDERED: float = 0
    PRICEEACH: float = 0
    MSRP: float = 0
    SALES: float = 0
    MONTH_ID: float = 0
    YEAR_ID: float = 0
    PRODUCTCODE: float = 0
    DAYS_SINCE_LASTORDER: float = 0
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
    Large: float = 0
    Medium: float = 0
    Small: float = 0

@app.get("/")
def inicio():
    return {"mensaje": "Microservicio activo - Acceso CORS habilitado"}

# --- RUTA CORREGIDA CON API_ROUTE PARA EVITAR ERROR 405 ---
@app.api_route("/predecir", methods=["GET", "POST", "OPTIONS"])
async def predecir(datos: Optional[DatosVenta] = None, request: Request = None):
    # Si la petición llega vacía o es un GET por error de redirección
    if request.method == "GET":
        return {"error": "Se recibió un GET. n8n debe enviar un POST a /predecir sin barra final."}
    
    try:
        # Convertimos los datos recibidos a una lista de valores
        cuerpo = await request.json()
        valores = list(cuerpo.values())
        
        # Validar que tengamos las 37 columnas
        if len(valores) != 37:
            return {"error": f"Se esperaban 37 columnas, llegaron {len(valores)}", "status": "Error"}

        X = np.array([valores])
        
        # Escalado y predicción
        datos_escalados = scaler.transform(X)
        grupo = kmeans.predict(datos_escalados)
        
        return {
            "segmento_venta": int(grupo[0]),
            "status": "Exitoso",
            "metodo_recibido": request.method
        }
    except Exception as e:
        return {"error": str(e), "status": "Fallido"}
