// Singleton de estado global: único objeto mutable compartido entre todos los módulos vía ES module instance.
// Todos los módulos importan este objeto y lo mutan directamente;
// los ES modules garantizan una sola instancia (singleton).

export const state = {
    // Datos del catálogo (poblados por catalog.js)
    catalog: { names: [], temps: [], radii: [], targets: [], ihps: [], score_ia: [], score_heller: [] },
    hoveredStarIndex: null,

    // Referencias a objetos Three.js de planetas (asignadas en catalog.js)
    pointsRef:      null,
    grialPointsRef: null,
    iaPointsRef:    null,

    // Puntero virtual unificado (mouse + mano); inicializado en scene.js como THREE.Vector2
    virtualPointer: null,

    // Control de rotación y navegación
    autoRotate:      true,
    autoRotateY:     0,
    isGrabbingSpace: false,
    targetQuaternion: null,   // THREE.Quaternion, inicializado en scene.js
    rotVelX: 0,
    rotVelY: 0,

    // Estado del hand tracking
    wasPinching:   false,
    lastZoomDist:  0,
    smoothTargetX: window.innerWidth  / 2,
    smoothTargetY: window.innerHeight / 2,

    // Gestión del ciclo de vida de recursos WebGL
    disposables:    [],
    animFrameId:    null,
    cameraUtilsRef: null,
    handsRef:       null,
};
