# Astrónomus - Índice de Habitabilidad Planetaria (IHP)
**Clasificación de Habitabilidad de Exoplanetas mediante Deep Learning**

![NASA API](https://img.shields.io/badge/Data-NASA_Exoplanet_Archive-112BBD?style=flat&logo=nasa)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat&logo=pytorch)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi)
![Gradio](https://img.shields.io/badge/Frontend-Gradio-FF7C00?style=flat&logo=gradio)

---

## Introducción y Dominio del Negocio
Este proyecto constituye el **Trabajo Práctico Obligatorio (TPO)** para la materia Ciencia de Datos.

## Arquitectura de la Solución
El proyecto fue diseñado bajo una arquitectura modular desacoplada (Backend/Frontend), garantizando escalabilidad y mantenibilidad.
```
planetary-habitability/
├── artifacts/          # Binarios (Modelo .pt, Scaler .pkl)
├── data/               # Caché local (Evita bloqueos de la API)
├── src/
│   ├── api/            # Controller: FastAPI & Frontend Gradio
│   ├── core/           # Service: Lógica física y pre-procesamiento
│   └── models/         # Entity: Red Neuronal y Motor de Entrenamiento
├── tests/              # Scripts de validación (Ingesta, Imputación, Target)
├── train.py            # Orquestador del entrenamiento
└── README.md
