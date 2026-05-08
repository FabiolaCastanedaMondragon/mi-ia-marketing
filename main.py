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
# Esto permite que n8n se conecte sin bloqueos de seguridad
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga de modelos (Asegúrate de que los nombres de archivo coincidan en tu repo)
try:
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    # autoencoder = tf.keras.models.load_model('modelo_ia.h5', compile=False) 
except Exception as e:
    print(f"Error al cargar modelos: {e}")

# Esquema de datos (37 columnas exactas)
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
    return {"mensaje": "Microservicio activo - FastAPI funcionando en Render"}

# --- RUTA MAESTRA: ACEPTA POST Y GET PARA EVITAR ERRORES DE REDIRECCIÓN ---
@app.api_route("/predecir", methods=["GET", "POST", "OPTIONS"])
async def predecir(request: Request):
    # Si n8n manda un GET por error, le avisamos amablemente
    if request.method == "GET":
        return {
            "error": "Se recibió un GET. n8n debe enviar un POST con JSON.",
            "url_correcta": "https://mi-ia-marketing.onrender.com/predecir",
            "instruccion": "No pongas '/' al final de la URL en n8n"
        }
    
    try:
        # Extraemos el JSON manualmente para mayor seguridad
        body = await request.json()
        
        # Convertimos el JSON en una lista ordenada de valores
        # Importante: El JSON en n8n debe tener las 37 llaves exactas
        valores = [
            body.get("QUANTITYORDERED", 0), body.get("PRICEEACH", 0), body.get("MSRP", 0),
            body.get("SALES", 0), body.get("MONTH_ID", 0), body.get("YEAR_ID", 0),
            body.get("PRODUCTCODE", 0), body.get("DAYS_SINCE_LASTORDER", 0),
            body.get("Australia", 0), body.get("Austria", 0), body.get("Belgium", 0),
            body.get("Canada", 0), body.get("Denmark", 0), body.get("Finland", 0),
            body.get("France", 0), body.get("Germany", 0), body.get("Ireland", 0),
            body.get("Italy", 0), body.get("Japan", 0), body.get("Norway", 0),
            body.get("Philippines", 0), body.get("Singapore", 0), body.get("Spain", 0),
            body.get("Sweden", 0), body.get("Switzerland", 0), body.get("UK", 0),
            body.get("USA", 0), body.get("Classic_Cars", 0), body.get("Motorcycles", 0),
            body.get("Planes", 0), body.get("Ships", 0), body.get("Trains", 0),
            body.get("Trucks_and_Buses", 0), body.get("Vintage_Cars", 0),
            body.get("Large", 0), body.get("Medium", 0), body.get("Small", 0)
        ]

        X = np.array([valores])
        
        # Proceso de IA
        datos_escalados = scaler.transform(X)
        grupo = kmeans.predict(datos_escalados)
        
        return {
            "segmento_venta": int(grupo[0]),
            "status": "Exitoso",
            "metodo": request.method
        }
    except Exception as e:
        return {"error": str(e), "status": "Fallido"}
