from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI(title="Predicción de Segmentos de Venta")

# Configuración de CORS (permite todo para evitar problemas con n8n)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelos
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')


@app.get("/")
def inicio():
    return {
        "mensaje": "Servidor funcionando correctamente ✅",
        "endpoint": "/predecir"
    }


@app.post("/predecir")
async def predecir(body: dict):
    try:
        claves = [
            "QUANTITYORDERED", "PRICEEACH", "MSRP", "SALES", "MONTH_ID", "YEAR_ID",
            "PRODUCTCODE", "DAYS_SINCE_LASTORDER", "Australia", "Austria", "Belgium",
            "Canada", "Denmark", "Finland", "France", "Germany", "Ireland", "Italy",
            "Japan", "Norway", "Philippines", "Singapore", "Spain", "Sweden",
            "Switzerland", "UK", "USA", "Classic_Cars", "Motorcycles", "Planes",
            "Ships", "Trains", "Trucks_and_Buses", "Vintage_Cars", "Large", "Medium", "Small"
        ]
       
        # Convertir los valores (0 por defecto si no existe la clave)
        valores = [float(body.get(k, 0)) for k in claves]
        
        X = np.array([valores])
        X_scaled = scaler.transform(X)
        grupo = kmeans.predict(X_scaled)
       
        return {
            "segmento_venta": int(grupo[0]),
            "status": "Exitoso"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "detalle": "Revisa que estés enviando todas las claves como números y en formato JSON correcto"
        }


# Opcional: Para aceptar GET también (útil para pruebas)
@app.get("/predecir")
async def predecir_get():
    return {"error": "Este endpoint solo acepta POST", "ayuda": "Usa POST desde n8n"}
