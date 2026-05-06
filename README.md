# ECG MLOps Service: Clasificación de Arritmias con InceptionTime

**Autor:** Ali Adib

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/)
[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFCC00?style=for-the-badge&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)

---

## 📋 Estado del Proyecto

| Estado | Descripción |
|--------|-------------|
| ✅ **API Desplegada** | Servicio activo en Render |
| 🚀 **Listo para Producción** | Endpoint funcional para usuarios |

---

## Arquitectura y Metodología

Este proyecto implementa un ecosistema completo de **MLOps** para la clasificación de señales de electrocardiograma (ECG):

| Componente | Tecnología | Descripción |
|------------|------------|-------------|
| **Entrenamiento** | PyTorch + WandB | Modelo InceptionTime con tracking |
| **Model Registry** | WandB Artifacts | Versionado (`model:v2`) |
| **API de Inferencia** | FastAPI | Servicio REST |
| **Contenerización** | Docker | Entorno reproducible |
| **Despliegue** | Render | Servicio cloud |
| **Testing** | Pytest | Validación |

---

## 🔗 Enlaces del Proyecto

### Producción
- **API Swagger UI:** https://ecg-mlops-project.onrender.com/docs  
- **Health Check:** https://ecg-mlops-project.onrender.com/health_check  
- **Endpoint de Predicción:** https://ecg-mlops-project.onrender.com/predict

### 📊 Weights & Biases
- **Reporte:**  
  https://wandb.ai/ali-adib-csic/ECG-MLOps-Project/reports/Reporte-de-MLOps-Comparativa-InceptionTime-vs-ROCKET-y-Trazabilidad-de-Artefactos--VmlldzoxNjY4NTMwNA  

- **Model Registry:**  
  https://wandb.ai/ali-adib-csic/ECG-MLOps-Project/artifacts  

### 💻 Código
- **GitHub:** https://github.com/aadib-csic/ECG-MLOps-Project

---

## 🚀 Uso en Producción

El servicio ya está desplegado y listo para usar.

👉 **El usuario NO necesita configurar nada**.

```text
Solo tienes que enviar una petición al endpoint /predict