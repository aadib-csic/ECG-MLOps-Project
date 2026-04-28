import pandas as pd
import kagglehub
import os
import shutil
from .utils import balance_data

class ECGDataLoader:
    def __init__(self, target_samples=20000, data_dir='/content/data'):
        self.target_samples = target_samples
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, 'mitbih_train.csv')
        self.test_path = os.path.join(data_dir, 'mitbih_test.csv')

    def download_data(self):
        """Descarga desde Kaggle y organiza en la carpeta local"""
        if not os.path.exists(self.train_path):
            print("Descargando dataset de Kaggle...")
            path = kagglehub.dataset_download("shayanfazeli/heartbeat")
            
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            
            shutil.copy(os.path.join(path, 'mitbih_train.csv'), self.train_path)
            shutil.copy(os.path.join(path, 'mitbih_test.csv'), self.test_path)
            print("Datos listos en:", self.data_dir)
        else:
            print("Los datos ya existen localmente.")

    def load_and_balance(self):
        """Carga los CSV y aplica el balanceo de clases"""
        train_df = pd.read_csv(self.train_path, header=None)
        test_df = pd.read_csv(self.test_path, header=None)
        
        print(f"Balanceando clases a {self.target_samples} muestras...")
        train_balanced = balance_data(train_df, n_samples=self.target_samples)
        
        return train_balanced, test_df
