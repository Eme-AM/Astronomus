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

El flujo de datos del proyecto sigue un paradigma **ELT (Extract, Load, Transform)** estructurado en tres capas para garantizar la trazabilidad y calidad de la información:

1. **Capa Bronce (Raw):** Ingesta cruda y directa a disco físico mediante `urllib` y `kagglehub`. Se preservan los metadatos y la varianza original sin intervención temprana de transformadores que puedan corromper formatos.
2. **Capa Plata (Processed):** Consolidación multi-fuente mediante resolución de entidades espaciales, imputación termodinámica y estadística, generando un único catálogo maestro de grado científico.
3. **Capa Oro (Artifacts):** Vectores y tensores matemáticamente escalados listos para su inyección en la Red Neuronal (*En desarrollo*).

---

## Fuentes de Datos Integradas

Para lograr una validación cruzada y enriquecer el espacio de características (*feature space*), el Data Lake ingesta dinámicamente tres repositorios:

*   **NASA Exoplanet Archive (`pscomppars`):** Utilizado como fuente primaria (Ground Truth). Aporta parámetros orbitales precisos (excentricidad, período) y datos estelares (edad, metalicidad).
*   **The Extrasolar Planets Encyclopaedia (Observatorio de París):** Utilizado para inyectar volumen y rellenar características físicas mediante fusión de datos (*Cross-filling*).
*   **Planetary Habitability Laboratory (Arecibo - PHL):** Extraído vía Kaggle API. Provee las etiquetas categóricas de validación externa precalculadas por astrofísicos: **Earth Similarity Index (ESI)** y estado de **Habitabilidad**.

---

## Pipeline de Procesamiento y Data Quality

### 1. Entity Resolution Espacial (Fuzzy Matching)
Debido a la carencia de una nomenclatura exoplanetaria universal (ej. *Kepler-186 f* vs *KOI-571.05*), se implementó un algoritmo de búsqueda espacial **K-D Tree** (`scipy.spatial.cKDTree`). Se utilizan las coordenadas celestes (Ascensión Recta y Declinación) como llave primaria universal, logrando cruces instantáneos con un radio de tolerancia microscópico ($0.01^\circ$).

### 2. Imputación Híbrida de 3 Niveles
Se rechazó la imputación por fuerza bruta (relleno con medianas globales) para evitar la alteración de la varianza astrofísica. En su lugar, se diseñó un rescate de datos en tres fases:
*   **Nivel 1 (Cross-Filling):** Inyección directa de observaciones empíricas cruzadas desde catálogos europeos.
*   **Nivel 2 (Termodinámica):** Deducción de Temperatura de Equilibrio e Insolación mediante derivaciones de la Ley de Stefan-Boltzmann ($T_{eq} = 255 \cdot S^{1/4}$) y cálculos de densidad planetoide.
*   **Nivel 3 (Machine Learning):** Implementación del algoritmo **MICE** (*Multiple Imputation by Chained Equations*) a través de un `IterativeImputer` impulsado por un ensamble de `ExtraTreesRegressor` para inferir variables complejas (ej. Edad Estelar) preservando correlaciones no lineales.

---

## Estructura del Repositorio

El código está estructurado bajo principios de mantenibilidad y modularidad:
```text
ASTRONOMUS/
├── artifacts/            # Modelos entrenados y escaladores (ignorados en git)
├── data/                 # Data Lake local
│   ├── raw/              # Capa Bronce: Descargas intocables
│   └── processed/        # Capa Plata: DataFrames consolidados
├── notebooks/            # Jupyter Notebooks para Análisis Exploratorio (EDA)
├── src/                  # Código fuente de la aplicación
│   ├── api/              # Endpoints para despliegue futuro (main_api.py)
│   ├── core/             # Fórmulas y leyes astrofísicas (physics.py)
│   ├── data/             # Módulos definitivos de ingesta (ingestion.py)
│   └── models/           # Arquitectura de la Red Neuronal (architecture.py)
├── tests/                # Scripts de prueba y experimentación (Multi/ y Simple/)
├── .gitignore            # Reglas de exclusión (data/, pycache, IDEs)
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Documentación principal
