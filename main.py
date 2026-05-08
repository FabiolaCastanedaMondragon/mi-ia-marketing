from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

# Configuración de CORS total
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga de modelos (asegúrate de que los nombres coincidan en tu repo)
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')

@app.get("/")
def inicio():
    return {"mensaje": "Servidor funcionando correctamente"}

# Esta ruta acepta TODO para que n8n no falle por redirecciones
@app.api_route("/predecir", methods=["GET", "POST", "OPTIONS"])
async def predecir(request: Request):
    if request.method == "GET":
        return {"error": "n8n debe usar POST", "ayuda": "Quita la / final de la URL en n8n"}
    
    try:
        body = await request.json()
        # Mapeo manual de las 37 columnas para evitar errores de n8n
        claves = [
            "QUANTITYORDERED", "PRICEEACH", "MSRP", "SALES", "MONTH_ID", "YEAR_ID", 
            "PRODUCTCODE", "DAYS_SINCE_LASTORDER", "Australia", "Austria", "Belgium", 
            "Canada", "Denmark", "Finland", "France", "Germany", "Ireland", "Italy", 
            "Japan", "Norway", "Philippines", "Singapore", "Spain", "Sweden", 
            "Switzerland", "UK", "USA", "Classic_Cars", "Motorcycles", "Planes", 
            "Ships", "Trains", "Trucks_and_Buses", "Vintage_Cars", "Large", "Medium", "Small"
        ]
        
        valores = [float(body.get(k, 0)) for k in claves]
        X = np.array([valores])
        X_scaled = scaler.transform(X)
        grupo = kmeans.predict(X_scaled)
        
        return {
            "segmento_venta": int(grupo[0]),
            "status": "Exitoso"
        }
    except Exception as e:
        return {"error": str(e), "detalle": "Asegúrate de enviar JSON con comillas dobles"}
