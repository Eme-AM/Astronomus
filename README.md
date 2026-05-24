# Astronomus: Deep Learning para la Clasificación de Exoplanetas

**Astronomus** es un proyecto integral de Ciencia e Ingeniería de Datos diseñado para predecir y clasificar la habitabilidad de exoplanetas. El proyecto combina astrofísica teórica, procesamiento espacial de datos y algoritmos avanzados de Machine Learning para consolidar y analizar información de los principales observatorios del mundo.

---

![NASA API](https://img.shields.io/badge/Data-NASA_Exoplanet_Archive-112BBD?style=flat&logo=nasa)
![Exoplanet.eu](https://img.shields.io/badge/Data-Observatoire_de_Paris-003366?style=flat&logo=european-space-agency)
![PHL Arecibo](https://img.shields.io/badge/Data-PHL_Arecibo_(Kaggle)-20BEFF?style=flat&logo=kaggle)

![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat&logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit_Learn-F7931E?style=flat&logo=scikit-learn)
![SciPy](https://img.shields.io/badge/Math-SciPy-8CAAEE?style=flat&logo=scipy)

---

## Arquitectura de Datos (Patrón Medallón)

El flujo de datos sigue un paradigma **ELT (Extract, Load, Transform)** estructurado en tres capas estrictas para garantizar la trazabilidad y sanidad de la información:

1. **Capa Bronce (`data/raw/`):** Ingesta cruda y directa a disco físico (*Import-Safe*). Se preservan los metadatos y la varianza original sin intervención temprana.
2. **Capa Plata (`data/silver/`):** Consolidación multi-fuente mediante resolución espacial 3D, imputación termodinámica y algoritmos de bosque aleatorio, generando un único catálogo maestro libre de valores nulos.
3. **Capa Oro (`data/gold/`):** Vectores matemáticos procesados. Incluye *Target Engineering* multiclase, partición estratificada y escalado robusto, listos para la Red Neuronal.
---

## Fuentes de Datos Integradas

Para lograr una validación cruzada y enriquecer el espacio de características (*feature space*), el Data Lake ingesta dinámicamente tres repositorios:

*   **NASA Exoplanet Archive (`pscomppars`):** Utilizado como fuente primaria (Ground Truth). Aporta parámetros orbitales precisos (excentricidad, período) y datos estelares (edad, metalicidad).
*   **The Extrasolar Planets Encyclopaedia (Observatorio de París):** Utilizado para inyectar volumen y rellenar características físicas mediante fusión de datos (*Cross-filling*).
*   **Planetary Habitability Laboratory (Arecibo - PHL):** Extraído vía Kaggle API. Provee las etiquetas categóricas de validación externa precalculadas por astrofísicos: **Earth Similarity Index (ESI)** y estado de **Habitabilidad**.

---

## Pipeline de Procesamiento y Data Quality

### 1. Entity Resolution Espacial (Fuzzy Matching)
Ante la carencia de una nomenclatura universal, se implementó un algoritmo **K-D Tree** (`scipy.spatial.cKDTree`). Para evitar la distorsión polar de las coordenadas celestes (Ascensión Recta y Declinación), se proyectan a vectores unitarios 3D utilizando la **Distancia de Cuerda**. El cruce resuelve colisiones dinámicamente mediante una política de *Closest-Wins*, asegurando la asignación del gemelo espacial más preciso dentro de un margen de $0.01^\circ$.

### 2. Imputación Híbrida de 3 Niveles
Se rechazó la imputación por fuerza bruta para evitar la alteración de la varianza astrofísica. El rescate de datos opera en tres fases:
* **Fase Empírica (Cross-Filling):** Inyección de observaciones cruzadas desde catálogos europeos.
* **Fase Determinista (Física):** Deducción exacta de la Temperatura de Equilibrio e Insolación mediante derivaciones de la Ley de Stefan-Boltzmann ($T_{eq} \propto S^{1/4}$).
* **Fase Estocástica (Machine Learning):** Algoritmo **MICE** (*Multiple Imputation by Chained Equations*) impulsado por un `ExtraTreesRegressor`, mitigando el sobreajuste mediante divisiones extremas aleatorias.

### 3. Target Engineering: El Súper Target
La clasificación de habitabilidad no se basa en un solo parámetro plano. Se construyó un objetivo multiclase combinando el estado de la Zona Habitable (*Goldilocks Zone*) y un exigente umbral en el *Earth Similarity Index* ($ESI \ge 0.80$), categorizando el universo en: `Mundo Inhóspito`, `Mundo Exótico` y el escaso `Tierra 2.0 (Grial)`.

---

## Estructura del Repositorio
```text
ASTRONOMUS/
├── artifacts/            # Modelos entrenados y escaladores (.pkl, .pth)
├── data/                 # Data Lake Local (Ignorado en versionado)
│   ├── archive/          # Históricos y snapshots para auditoría
│   ├── raw/              # Capa Bronce: Descargas intocables
│   ├── silver/           # Capa Plata: Data Lake consolidado
│   └── gold/             # Capa Oro: Tensores escalados y estratificados
├── reports/
│   └── figures/          # Gráficos de calidad de publicación (EDA, PCA, KDE)
├── src/
│   ├── api/              # Endpoints para despliegue futuro (FastAPI)
│   ├── core/             # Fórmulas y leyes astrofísicas (physics.py)
│   ├── data/             # Pipeline ELT (ingestion.py, processing.py, preparation.py)
│   ├── models/           # Topología de la Red Neuronal Profunda (architecture.py)
│   └── visualization/    # Suite de scripts de diagnóstico y auditoría MLOps
├── tests/                # Scripts de prueba y experimentación
├── .gitignore            # Reglas de exclusión (Data, Artifacts, PyCache)
├── requirements.txt      # Dependencias
└── README.md             # Documentación
