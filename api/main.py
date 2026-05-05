import os
import torch
import numpy as np
import wandb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Importar arquitectura y preprocesamiento de la carpeta src
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
    # El usuario envía una lista de 187 valores
    signal: List[float]

@app.on_event("startup")
def load_candidate_model():
    """Descarga el artefacto v2 de W&B y reconstruye el modelo usando API directa (sin sockets)."""
    global model_engine
    try:
        # Configuración para entorno cloud (desactiva comunicación por sockets)
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_SILENT"] = "true"
        
        # 1. Autenticación explícita para entornos en la nube (Render)
        api_key = os.environ.get("WANDB_API_KEY") or os.getenv("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)
            print("✅ Autenticación W&B exitosa")
        else:
            print("❌ WANDB_API_KEY no encontrada en variables de entorno")
            return

        print(f"Descargando artefacto: {MODEL_ARTIFACT}...")
        print("DEBUG ENV KEYS:", list(os.environ.keys()))
        
        # Usar API directa en lugar de wandb.init()
        api = wandb.Api()
        artifact = api.artifact(MODEL_ARTIFACT)
        artifact_dir = artifact.download()
        print(f"✅ Artefacto descargado en: {artifact_dir}")
        
        # 2. Instanciar la arquitectura
        model_engine = InceptionTime(n_classes=5, nf=16)
        
        # 3. Buscar y cargar los pesos
        import glob
        weight_files = glob.glob(os.path.join(artifact_dir, "*.pth")) + glob.glob(os.path.join(artifact_dir, "*.pt"))
        path_weights = weight_files[0] if weight_files else None
        
        if path_weights:
            print(f"Cargando pesos desde: {os.path.basename(path_weights)}")
            state_dict = torch.load(path_weights, map_location='cpu')
            model_engine.load_state_dict(state_dict)
            model_engine.eval()
            print(f"✅ Modelo cargado exitosamente")
        else:
            raise FileNotFoundError("No se encontró un archivo de pesos (.pth o .pt) en el artefacto.")

    except Exception as e:
        # Imprimir el error en los Logs de Render
        print(f"❌ ERROR CRÍTICO al cargar modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        model_engine = None

@app.get("/health_check")
def health_check():
    return {
        "status": "online" if model_engine else "model_error", 
        "model": MODEL_ARTIFACT,
        "worker": "uvicorn"
    }

@app.post("/predict")
async def predict(request: ECGRequest):
    if model_engine is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible. Revisa los logs de inicio del servidor."
        )

    try:
        # 1. Preparar datos para el preprocesador
        data_list = request.signal
        if len(data_list) < 187:
            data_list = data_list + [0.0] * (187 - len(data_list))
            
        temp_df = pd.DataFrame([data_list])
        
        # 2. Preprocesar (X tendrá forma 1, 1, 187)
        X, _ = preprocess_signal(temp_df)
        input_tensor = torch.from_numpy(X).float()

        # 3. Inferencia
        with torch.no_grad():
            output = model_engine(input_tensor)
            
            # Manejar si la salida es tupla o tensor directo
            logits = output if isinstance(output, torch.Tensor) else output[0]
            
            prob = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(prob, dim=1).item()
            confidence = prob[0][pred_class].item()

        # Mapeo de clases del MIT-BIH
        classes = {0: "Normal", 1: "S (Supraventricular)", 2: "V (Ventricular)", 3: "F (Fusion)", 4: "Q (Unknown)"}

        return {
            "prediction": classes.get(pred_class, "Unknown"),
            "class_index": pred_class,
            "confidence": round(confidence, 4),
            "input_length": len(request.signal)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {str(e)}")