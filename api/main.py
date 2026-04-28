import os
import torch
import numpy as np
import wandb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Importamos tu arquitectura y preprocesamiento de la carpeta src
from src.model import InceptionTime
from src.utils import preprocess_signal

app = FastAPI(
    title="ECG MLOps Service",
    description="Servicio de clasificación de arritmias cardíacas usando InceptionTime"
)

# --- CONFIGURACIÓN DE W&B ---
ENTITY = "ali-adib-csic"
PROJECT = "ECG-MLOps-Project"
MODEL_ARTIFACT = f"{ENTITY}/{PROJECT}/model:v2"

# Variable global para el modelo
model_engine = None

class ECGRequest(BaseModel):
    # El usuario envía una lista de 187 valores (longitud estándar del MIT-BIH)
    signal: List[float]

@app.on_event("startup")
def load_candidate_model():
    """Descarga el artefacto v2 de W&B y reconstruye el modelo."""
    global model_engine
    try:
        print(f"Descargando artefacto: {MODEL_ARTIFACT}...")
        run = wandb.init(project=PROJECT, job_type="inference")
        artifact = run.use_artifact(MODEL_ARTIFACT, type='model')
        artifact_dir = artifact.download()
        
        # 1. Instanciar la arquitectura (nf=64 según tu default_config en train.py)
        # Nota: Si el nf cambió en el sweep, se podría leer de artifact.metadata
        model_engine = InceptionTime(n_classes=5, nf=64)
        
        # 2. Cargar los pesos (el archivo suele llamarse best_model.pth o model.pth)
        # Buscamos el archivo .pth en el directorio descargado
        path_weights = next(iter([f for f in os.listdir(artifact_dir) if f.endswith('.pth')]), None)
        
        if path_weights:
            state_dict = torch.load(os.path.join(artifact_dir, path_weights), map_location='cpu')
            model_engine.load_state_dict(state_dict)
            model_engine.eval()
            print("✅ Modelo cargado y listo para inferencia.")
        else:
            raise FileNotFoundError("No se encontró archivo .pth en el artefacto.")
            
        run.finish()
    except Exception as e:
        print(f"❌ Error crítico al cargar modelo: {e}")

@app.get("/")
def health_check():
    return {"status": "online", "model": MODEL_ARTIFACT}

@app.post("/predict")
async def predict(request: ECGRequest):
    if model_engine is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    try:
        # 1. Convertir entrada a DataFrame temporal para reusar tu preprocess_signal
        # Tu función espera un DF con la etiqueta al final, añadimos un 0 ficticio
        import pandas as pd
        temp_df = pd.DataFrame([request.signal + [0]])
        
        # 2. Preprocesar (X tendrá forma 1, 1, 187)
        X, _ = preprocess_signal(temp_df)
        input_tensor = torch.from_numpy(X).float()

        # 3. Inferencia
        with torch.no_grad():
            output = model_engine(input_tensor)
            # InceptionTime devuelve (fc_output, ), tomamos el primer elemento
            logits = output if isinstance(output, torch.Tensor) else output[0]
            
            prob = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(prob, dim=1).item()
            confidence = prob[0][pred_class].item()

        # Mapeo de clases del MIT-BIH
        classes = {0: "Normal", 1: "S", 2: "V", 3: "F", 4: "Q"}

        return {
            "prediction": classes.get(pred_class, "Unknown"),
            "class_index": pred_class,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {str(e)}")
