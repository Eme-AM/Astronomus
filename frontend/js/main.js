import { initScene, renderer, cleanupAll } from './scene.js';
import { loadCatalog }      from './catalog.js';
import { animate, initInteraction } from './interaction.js';
import { initWebcamAndAI }  from './handtracking.js';

/*
Punto de entrada: inicializa los módulos en orden y 
registra listeners globales de ciclo de vida.
*/
initScene();

// El handler de contexto restaurado necesita animate, que solo está disponible
// después de que interaction.js está cargado, por eso se registra aquí.
renderer.domElement.addEventListener('webglcontextrestored', animate);
window.addEventListener('beforeunload', cleanupAll);

initInteraction();
loadCatalog();
initWebcamAndAI();
animate();
