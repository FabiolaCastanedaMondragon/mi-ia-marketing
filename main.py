from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI(title="Predicción de Segmentos de Venta")

# CORS
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
       
        valores = []
        for k in claves:
            valor = body.get(k)
            
            if k == "PRODUCTCODE":
                # Convertir a string normal
                valores.append(str(valor).strip() if valor is not None else "")
            else:
                # Convertir a float de forma segura
                try:
                    valores.append(float(valor) if valor is not None else 0.0)
                except (ValueError, TypeError):
                    valores.append(0.0)
        
        # Convertir a array numpy
        X = np.array([valores], dtype=object)
        
        # Convertir solo las columnas numéricas para el scaler
        X_numeric = np.array([valores[0:7] + valores[8:]], dtype=float)  # excluyendo PRODUCTCODE
        
        X_scaled = scaler.transform(X_numeric)
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
    return {"error": "Este endpoint solo acepta POST", "ayuda": "Usa POST desde n8n"}
