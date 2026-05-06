import numpy as np
import pandas as pd
from sklearn.utils import resample

def preprocess_signal(df):
    """
    Convierte el DataFrame en tensores de PyTorch
    """
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X.astype(np.float32), y.astype(np.int64)

def balance_data(df, n_samples=20000):
    """
    Aplica el balanceo de clases que usaste en tu práctica
    """
    df_0 = df[df[187] == 0]
    df_1 = df[df[187] == 1]
    df_2 = df[df[187] == 2]
    df_3 = df[df[187] == 3]
    df_4 = df[df[187] == 4]

    # Remuestreo para equilibrar las clases a n_samples
    df_0_res = resample(df_0, replace=True, n_samples=n_samples, random_state=42)
    df_1_res = resample(df_1, replace=True, n_samples=n_samples, random_state=42)
    df_2_res = resample(df_2, replace=True, n_samples=n_samples, random_state=42)
    df_3_res = resample(df_3, replace=True, n_samples=n_samples, random_state=42)
    df_4_res = resample(df_4, replace=True, n_samples=n_samples, random_state=42)

    return pd.concat([df_0_res, df_1_res, df_2_res, df_3_res, df_4_res])
