# ECG MLOps Service: Clasificación de Arritmias con InceptionTime
**Autor:** Ali Adib

Este proyecto despliega un ecosistema de **MLOps** para la clasificación de señales de electrocardiograma (ECG) en tiempo real. 

## Arquitectura y Metodología
El flujo de trabajo sigue los estándares de la industria para garantizar la portabilidad y escalabilidad del modelo:

*   **Entrenamiento:** Basado en la arquitectura InceptionTime, con seguimiento de métricas y experimentos en **Weights & Biases**.
*   **Model Registry:** Gestión de versiones mediante **WandB Artifacts**. La API descarga dinámicamente el mejor modelo (`candidate`) al iniciar.
*   **Servicio de Inferencia:** Backend construido con **FastAPI** para ofrecer predicciones de baja latencia.
*   **Contenerización:** Uso de **Docker** para asegurar que el servicio funcione de forma idéntica en cualquier entorno.
*   **Despliegue (Production):** Alojado en **Render** con un endpoint público accesible para su consumo.

## Validación y Calidad
*   **Testing:** El proyecto incluye una suite de pruebas en el directorio `/tests` utilizando **Pytest** para verificar la integridad de la API y el modelo.

*   **Trazabilidad:** Se utiliza el registro de WandB.

## Instalación y Uso Local

### 1. Clonar el repositorio
```bash
git clone [https://github.com/aadib-csic/ECG-MLOps-Project.git](https://github.com/aadib-csic/ECG-MLOps-Project.git)
cd ECG-MLOps-Project

## 🔗 Enlaces del Proyecto


- **API en Producción (Swagger UI):**  
  [Abrir Swagger](https://ecg-mlops-project.onrender.com/docs#/default/health_check__get)

- **Model Registry y Reporte (Weights & Biases):**  
  [Ver reporte en W&B](https://wandb.ai/ali-adib-csic/ECG-MLOps-Project/reports/Reporte-de-MLOps-Comparativa-InceptionTime-vs-ROCKET-y-Trazabilidad-de-Artefactos--VmlldzoxNjY4NTMwNA)

- **Repositorio GitHub:**  
  [Ver código](https://github.com/aadib-csic/ECG-MLOps-Project.git)

## Instalación y Uso Local
```bash

### 1. Clonar el repositorio
```bash
git clone https://github.com/aadib-csic/ECG-MLOps-Project.git
cd ECG-MLOps-Project

# 2. Construir la imagen
docker build -t ecg-mlops-service .

# 3. Ejecutar contenedor
docker run -p 8000:8000 ecg-mlops-service