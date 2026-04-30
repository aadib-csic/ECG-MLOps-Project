# ECG MLOps Service: Clasificación de Arritmias con InceptionTime

Este proyecto despliega un ecosistema de **MLOps** para la clasificación de señales de electrocardiograma (ECG) en tiempo real. El sistema integra el entrenamiento experimental, el registro de modelos y el despliegue productivo.

##  Arquitectura del Sistema
El flujo de trabajo sigue los principios de entrega continua para Machine Learning:
*   **Entrenamiento y Tracking:** Experimentos realizados en Google Colab con seguimiento de métricas en **Weights & Biases**.
*   **Model Registry:** Gestión de versiones mediante **WandB Artifacts**. La API descarga la versión `model:v2` al iniciar.
*   **Backend:** API de alto rendimiento construida con **FastAPI**.
*   **Contenerización:** Empaquetado con **Docker** para garantizar la paridad entre entornos.
*   **Despliegue:** Alojado en **Render** con pipeline de construcción automática.

## 📂 Gestión de Datos y Modelos (Versioning)
Para cumplir con los estándares de reproducibilidad, se han seguido dos estrategias:
1.  **Versionado de Modelos:** Uso nativo de **WandB** para el ciclo de vida del modelo.
2.  **Versionado de Datos (DVC):** El repositorio está preparado con **DVC (Data Version Control)** para el seguimiento de los datasets. Los datos pesados se mantienen fuera de la imagen de Docker para optimizar el despliegue, vinculando la trazabilidad al entorno de entrenamiento.

##  Validación y Calidad (CI/CD)
Se ha implementado un "Quality Gate" en el proceso de construcción de Docker:
*   **Pytest:** Ejecución automática de tests unitarios durante el `docker build`. Si los tests fallan, el despliegue se cancela.
*   **Integración:** Verificación de la carga del modelo y la estructura de respuesta de la API.

## 🔗 Enlaces del Proyecto
*   **Endpoint de Producción (Swagger UI):** [https://ecg-mlops-project.onrender.com/docs](https://ecg-mlops-project.onrender.com/docs)
*   **Model Registry (WandB):** [https://ecg-mlops-project.onrender.com/docs#/default/health_check__get]
*   **Repositorio GitHub:** [https://github.com/aadib-csic/ECG-MLOps-Project.git]

## Instalación y Uso Local
```bash
# 1. Construir la imagen
docker build -t ecg-mlops-service .

# 2. Ejecutar contenedor
docker run -p 8000:8000 ecg-mlops-service