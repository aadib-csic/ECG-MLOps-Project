import wandb
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sktime.transformations.panel.rocket import Rocket
from pathlib import Path
import sys

# Configuración de rutas
root = Path("/content/drive/MyDrive/Proyecto_MLOps_ECG").resolve()
sys.path.append(str(root))

from src.data_loader import ECGDataLoader
from src.utils import preprocess_signal

def train_rocket():
    # Inicializar wandb para el Baseline
    run = wandb.init(project="ECG-MLOps-Project", job_type="baseline")
    
    # Configuración fija para el Baseline
    config = {
        "num_kernels": 1000, # ROCKET usa kernels aleatorios
        "target_samples": 500 # lo mismo que el sweep rápido para comparar
    }
    wandb.config.update(config)

    # 1. Carga de datos
    loader = ECGDataLoader(target_samples=config["target_samples"])
    loader.download_data()
    train_df, test_df = loader.load_and_balance()

    X_train, y_train = preprocess_signal(train_df)
    X_test, y_test = preprocess_signal(test_df)

    # ROCKET espera (n_instances, n_columns, n_timepoints)
    # señales ya tienen esa forma (N, 1, 187)
    
    print(f"🚀 Iniciando ROCKET con {config['num_kernels']} kernels...")
    
    # 2. Transformación ROCKET
    rocket = Rocket(num_kernels=config["num_kernels"])
    rocket.fit(X_train)
    X_train_transform = rocket.transform(X_train)
    
    # 3. Clasificador 
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transform, y_train)

    # 4. Evaluación
    X_test_transform = rocket.transform(X_test)
    accuracy = classifier.score(X_test_transform, y_test)
    
    print(f"✅ ROCKET Accuracy: {accuracy:.4f}")

    # 5. Log a W&B
    wandb.log({"accuracy": accuracy, "model_type": "ROCKET_Baseline"})
    run.finish()

if __name__ == "__main__":
    train_rocket()
