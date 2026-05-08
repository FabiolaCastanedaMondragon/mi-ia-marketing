from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI(title="Predicción de Segmentos de Venta")

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
    return {"mensaje": "✅ Servidor funcionando correctamente"}


@app.post("/predecir")
async def predecir(body: dict):
    try:
        # 37 features que espera el scaler
        claves = [
            "QUANTITYORDERED", "PRICEEACH", "MSRP", "SALES", "MONTH_ID", "YEAR_ID",
            "PRODUCTCODE", "DAYS_SINCE_LASTORDER", "Australia", "Austria", "Belgium",
            "Canada", "Denmark", "Finland", "France", "Germany", "Ireland", "Italy",
            "Japan", "Norway", "Philippines", "Singapore", "Spain", "Sweden",
            "Switzerland", "UK", "USA", "Classic_Cars", "Motorcycles", "Planes",
            "Ships", "Trains", "Trucks_and_Buses", "Vintage_Cars", "Large", "Medium", "Small"
        ]
       
        valores = []
        for k in claves:
            valor = body.get(k)
            if k == "PRODUCTCODE":
                # Convertimos el código a número (hash simple) o 0
                try:
                    valores.append(float(hash(str(valor)) % 10000))  
                except:
                    valores.append(0.0)
            else:
                try:
                    valores.append(float(valor) if valor is not None else 0.0)
                except (ValueError, TypeError):
                    valores.append(0.0)

        X = np.array([valores], dtype=float)
        X_scaled = scaler.transform(X)
        grupo = kmeans.predict(X_scaled)
       
        return {
            "segmento_venta": int(grupo[0]),
            "status": "Exitoso"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "detalle": "Error al procesar los datos"
        }


@app.get("/predecir")
async def predecir_get():
    return {"error": "Solo POST"}
