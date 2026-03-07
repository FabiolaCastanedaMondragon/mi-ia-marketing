from fastapi import FastAPI
import joblib
import pandas as pd
import tensorflow as tf
import uvicorn

app = FastAPI()

# Cargamos los cerebros que descargaste
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')
model = tf.keras.models.load_model('modelo_ia.h5', compile=False)

@app.get("/")
def inicio():
    return {"mensaje": "El microservicio de Marketing IA está funcionando"}

@app.post("/predecir")
def predecir(datos: dict):
    # Convertimos los datos que nos manden en una tabla (DataFrame)
    df = pd.DataFrame([datos])
    
    # Escalamos los datos como hacías en el notebook
    datos_escalados = scaler.transform(df)
    
    # Predicción del grupo (KMeans)
    grupo = kmeans.predict(datos_escalados)
    
    # Predicción de la Red Neuronal
    prediccion_ia = model.predict(datos_escalados)
    
    return {
        "segmento_cliente": int(grupo[0]),
        "resultado_ia": prediccion_ia.tolist()
    }
