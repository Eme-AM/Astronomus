# Astronomus: Deep Learning para la Clasificación de Exoplanetas

**Astronomus** es un proyecto integral de Ciencia e Ingeniería de Datos diseñado para predecir y clasificar la habitabilidad de exoplanetas. El proyecto combina astrofísica teórica, procesamiento espacial de datos y algoritmos avanzados de Machine Learning para consolidar y analizar información de los principales observatorios del mundo, y lo expone mediante un visor 3D interactivo controlado por gestos de mano en tiempo real.

---

![NASA API](https://img.shields.io/badge/Data-NASA_Exoplanet_Archive-112BBD?style=flat&logo=nasa)
![Exoplanet.eu](https://img.shields.io/badge/Data-Observatoire_de_Paris-003366?style=flat&logo=european-space-agency)
![PHL Arecibo](https://img.shields.io/badge/Data-PHL_Arecibo_(Kaggle)-20BEFF?style=flat&logo=kaggle)

![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat&logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit_Learn-F7931E?style=flat&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat&logo=fastapi)
![Three.js](https://img.shields.io/badge/3D-Three.js_r162-black?style=flat&logo=three.js)

---

## Cómo ejecutar el proyecto

El orquestador principal está en la raíz del repositorio:

```bash
python main.py
```

Se presenta un menú interactivo para elegir qué parte ejecutar. Para usuarios avanzados, se pueden usar flags directamente:

| Flag | Acción |
|---|---|
| `--all` | Ejecuta el pipeline de datos completo (Bronce → Plata → Oro) |
| `--ingest` | Solo ingesta de datos (Capa Bronce) |
| `--process` | Solo procesamiento y consolidación (Capa Plata) |
| `--prepare` | Solo preparación de tensores (Capa Oro) |
| `--serve` | Levanta el servidor FastAPI + visor 3D en `http://localhost:8000` |

```bash
# Ejemplos
python main.py --all    # Construir el Data Lake desde cero
python main.py --serve  # Lanzar la interfaz 3D
```

---

## Arquitectura de Datos (Patrón Medallón)

El flujo de datos sigue un paradigma **ELT (Extract, Load, Transform)** estructurado en tres capas estrictas para garantizar la trazabilidad y sanidad de la información:

1. **Capa Bronce (`backend/data/raw/`):** Ingesta cruda y directa a disco físico (*Import-Safe*). Se preservan los metadatos y la varianza original sin intervención temprana.
2. **Capa Plata (`backend/data/silver/`):** Consolidación multi-fuente mediante resolución espacial 3D, imputación termodinámica y algoritmos de bosque aleatorio, generando un único catálogo maestro libre de valores nulos.
3. **Capa Oro (`backend/data/gold/`):** Vectores matemáticos procesados. Incluye *Target Engineering* multiclase, partición estratificada y escalado robusto, listos para la Red Neuronal.

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
La clasificación de habitabilidad no se basa en un solo parámetro plano. Se construyó un objetivo multiclase combinando el estado de la Zona Habitable (*Goldilocks Zone*) y un exigente umbral en el *Earth Similarity Index* ($ESI \ge 0.80$), categorizando el universo en cuatro clases:

| Clase | Etiqueta | Descripción |
|---|---|---|
| `0` | Mundo Inhóspito | Condiciones extremas, gigante gaseoso o roca estéril |
| `1` | Mundo Exótico | Composición anómala o atmósfera densa inusual |
| `2` | Tierra 2.0 (Grial) | Zona habitable + ESI ≥ 0.80, candidato a agua líquida |
| `3` | Hallazgo IA | Alto IHP detectado por el modelo, pendiente de confirmación |

El **Índice de Habitabilidad del Planeta (IHP)** se calcula combinando el score del modelo de anomalías (`score_ia`) con el score del índice de Heller (`score_heller`) para producir una métrica de confianza unificada.

---

## Visor 3D Interactivo

El frontend es una aplicación WebGL en tiempo real que visualiza el catálogo completo de exoplanetas como una esfera navegable en el espacio. Se controla con gestos de mano vía webcam (MediaPipe Hands) o con el cursor.

**Gestos disponibles:**

| Gesto | Acción |
|---|---|
| Puño cerrado | Rotar la esfera 3D en cualquier dirección |
| Índice extendido | Apuntar y hacer hover sobre planetas |
| Pulgar + dedo medio (1 mano) | Seleccionar planeta apuntado |
| Pulgar + índice (2ª mano) | Seleccionar planeta apuntado |
| 2 manos abiertas + separar | Zoom |

**Características técnicas del frontend:**

- Three.js r162 vía ES Modules + importmap (sin bundler)
- Rotación por quaterniones con `premultiply` en ejes del mundo → sin gimbal lock
- Inercia post-agarre con damping (coasting suave al soltar la mano)
- Auto-rotación que se reanuda automáticamente tras 3 segundos sin gestos
- Bloom post-procesado (EffectComposer + UnrealBloomPass)
- Frame skipping adaptivo: procesa 1/3 de frames cuando no hay manos detectadas
- Objetos THREE cacheados a nivel de módulo para evitar GC pressure a 60fps
- MediaPipe Hands con `modelComplexity: 1` para mayor precisión en gestos finos

---

## Estructura del Repositorio

```text
ASTRONOMUS/
├── main.py               # Orquestador principal (menú interactivo y flags CLI)
├── frontend/             # Visor 3D interactivo (WebGL + MediaPipe)
│   ├── index.html        # Shell HTML con importmap para Three.js
│   ├── css/
│   │   └── astronomus.css
│   └── js/
│       ├── main.js       # Punto de entrada: inicializa módulos en orden
│       ├── state.js      # Singleton de estado global compartido
│       ├── scene.js      # Renderer, cámara, bloom y controles orbitales
│       ├── catalog.js    # Carga el catálogo desde la API y crea los puntos 3D
│       ├── interaction.js # Raycasting, animación y listeners de mouse
│       └── handtracking.js # MediaPipe Hands: gestos de rotar, apuntar, zoom y clic
├── backend/
│   ├── main.py           # Alias de ejecución (importado por el orquestador raíz)
│   ├── requirements.txt  # Dependencias Python
│   ├── data/             # Data Lake Local (ignorado en versionado)
│   │   ├── raw/          # Capa Bronce: descargas intocables
│   │   ├── silver/       # Capa Plata: catálogo consolidado
│   │   └── gold/         # Capa Oro: tensores escalados y estratificados
│   ├── artifacts/        # Modelos entrenados y escaladores (.pkl, .pth)
│   ├── reports/
│   │   └── figures/      # Gráficos de calidad (EDA, métricas, matrices de confusión)
│   └── src/
│       ├── api/          # FastAPI: endpoints REST + servidor del frontend
│       │   └── routers/
│       │       └── exoplanets.py  # /api/catalog y endpoints de inferencia
│       ├── core/         # Fórmulas y leyes astrofísicas (physics.py)
│       ├── data/         # Pipeline ELT (ingestion.py, processing.py, preparation.py)
│       ├── models/       # Autoencoder de anomalías y scripts de entrenamiento
│       └── visualization/ # Suite de diagnóstico y auditoría MLOps
├── docs/
│   └── FRONTEND_REFACTORING.md  # Detalle técnico del frontend vs. commit de referencia
├── .gitignore
└── README.md
```

---

## Dependencias principales

```bash
pip install -r backend/requirements.txt
```

El frontend no tiene dependencias locales: Three.js y MediaPipe se cargan desde CDN en el navegador.
