from fastapi import FastAPI
import joblib
import pandas as pd
import tensorflow as tf
from pydantic import BaseModel

app = FastAPI()

# Cargamos los archivos
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')
model = tf.keras.models.load_model('modelo_ia.h5', compile=False)

# Definimos la estructura exacta para que no haya errores
class Cliente(BaseModel):
    BALANCE: float
    BALANCE_FREQUENCY: float
    PURCHASES: float
    ONEOFF_PURCHASES: float
    INSTALLMENTS_PURCHASES: float
    CASH_ADVANCE: float
    PURCHASES_FREQUENCY: float
    ONEOFF_PURCHASES_FREQUENCY: float
    PURCHASES_INSTALLMENTS_FREQUENCY: float
    CASH_ADVANCE_FREQUENCY: float
    CASH_ADVANCE_TRX: int
    PURCHASES_TRX: int
    CREDIT_LIMIT: float
    PAYMENTS: float
    MINIMUM_PAYMENTS: float
    PRC_FULL_PAYMENT: float
    TENURE: int

@app.get("/")
def inicio():
    return {"mensaje": "El microservicio de Marketing IA está funcionando"}

@app.post("/predecir")
def predecir(cliente: Cliente):
    try:
        # Convertimos el objeto cliente a diccionario y luego a DataFrame
        df = pd.DataFrame([cliente.dict()])
        
        # 1. Escalamos los datos
        datos_escalados = scaler.transform(df)
        
        # 2. Predicción del grupo (KMeans)
        grupo = kmeans.predict(datos_escalados)
        
        return {
            "segmento_cliente": int(grupo[0]),
            "status": "Exitoso"
        }
    except Exception as e:
        return {"error": str(e)}
