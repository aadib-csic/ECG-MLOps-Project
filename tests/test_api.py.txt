from fastapi.testclient import TestClient
import sys
import os

# Asegura que Python encuentre la carpeta 'api' y 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

def test_read_root():
    """Verifica que el endpoint raíz responda correctamente"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_prediction_endpoint_exists():
    """Verifica que el endpoint de predicción esté disponible"""
    
    response = client.post("/predict", json={}) 
    # Esperar un 422 (Unprocessable Entity) porque enviamos JSON vacío, 
    # lo cual confirma que el endpoint existe y pide datos.
    assert response.status_code == 422