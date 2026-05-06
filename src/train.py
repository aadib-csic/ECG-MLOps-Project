import wandb
import torch
import sys
import numpy as np
from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback
from pathlib import Path

# Configuración de rutas
root = Path("/content/drive/MyDrive/Proyecto_MLOps_ECG").resolve()
sys.path.append(str(root))

from src.model import InceptionTime
from src.data_loader import ECGDataLoader
from src.utils import preprocess_signal

def train_inception(config=None):
    # Inicializamos wandb
    if config is None:
        config = {}
    
    run = wandb.init(config=config)
    config = wandb.config
    
    print("Configuración:", dict(config))
    
    # Carga de datos
    print("Cargando datos...")
    loader = ECGDataLoader(target_samples=int(config.target_samples))
    loader.download_data() 
    train_df, test_df = loader.load_and_balance()
    
    print("Preprocesando...")
    X_train, y_train = preprocess_signal(train_df)
    X_test, y_test = preprocess_signal(test_df)
    
    print(f"Shape de datos: Train={X_train.shape}, Test={X_test.shape}")
    
    # Convertir a tensores y crear datasets para fastai
    # Fastai puede trabajar directamente con tensores de PyTorch
    train_x = torch.FloatTensor(X_train)
    train_y = torch.LongTensor(y_train)
    test_x = torch.FloatTensor(X_test)
    test_y = torch.LongTensor(y_test)
    
    # Crear DataLoaders usando el método de fastai
    
    class CustomDataset:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]
    
    train_ds = CustomDataset(train_x, train_y)
    test_ds = CustomDataset(test_x, test_y)
    
    # DataLoaders
    dls = DataLoaders.from_dsets(
        train_ds, test_ds, 
        bs=int(config.batch_size),
        num_workers=0
    )
    
    if torch.cuda.is_available():
        dls.cuda()
        print("Usando GPU")
    else:
        print("Usando CPU")
    
    # Modelo
    model = InceptionTime(n_classes=5, nf=int(config.nf))
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Learner con WandbCallback
    learn = Learner(
        dls, 
        model, 
        loss_func=LabelSmoothingCrossEntropy(), 
        metrics=[accuracy],
        cbs=[WandbCallback(log_preds=False, log_model=True)]
    )
    
    # Entrenar
    print(f"Iniciando entrenamiento por {config.epochs} épocas...")
    learn.fit_one_cycle(int(config.epochs), float(config.lr))
    
    # Guardar modelo
    model_path = Path("/content/drive/MyDrive/Proyecto_MLOps_ECG/models/best_model.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")
    
    # Guardar modelo en wandb
    wandb.save(str(model_path))
    
    # Mostrar resultados finales
    if len(learn.recorder.values) > 0:
        final_metrics = learn.recorder.values[-1]
        print(f"\n📊 Resultados finales:")
        print(f"  - Valid Loss: {final_metrics[0]:.4f}")
        print(f"  - Accuracy: {final_metrics[1]:.4f}")
    
    run.finish()
    print("✅ Entrenamiento completado!")

if __name__ == "__main__":
    # Configuración por defecto
    default_config = {
        "epochs": 10,
        "batch_size": 32,
        "lr": 0.001,
        "nf": 64,
        "target_samples": 500
    }
    train_inception(default_config)
