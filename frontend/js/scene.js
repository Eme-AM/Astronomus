import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass }     from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { state } from './state.js';

/*
Motor de escena: crea y exporta renderer, cámara, spaceGroup,
bloom (EffectComposer) y controles orbitales.
*/

// Exportaciones con let para que main.js reciba el valor vivo tras initScene()
export let renderer, scene, camera, spaceGroup, stardust, composer, controls, reticle;

export function initScene() {
    state.virtualPointer  = new THREE.Vector2(-999, -999);
    state.targetQuaternion = new THREE.Quaternion();

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(innerWidth, innerHeight);
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    document.body.appendChild(renderer.domElement);

    renderer.domElement.addEventListener('webglcontextlost', (e) => {
        e.preventDefault();
        cancelAnimationFrame(state.animFrameId);
    });

    // Escena y cámara
    scene  = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 2000);
    camera.position.set(0, 40, 220);

    // Bóveda celeste (wireframe de referencia)
    scene.add(new THREE.Mesh(
        new THREE.SphereGeometry(200, 100, 100),
        new THREE.MeshBasicMaterial({ color: 0x0a0a14, wireframe: true, transparent: true, opacity: 0.07 })
    ));

    spaceGroup = new THREE.Group();
    scene.add(spaceGroup);

    // Polvo estelar de fondo
    const stardustCount = 3000;
    const stardustPos   = new Float32Array(stardustCount * 3);
    for (let i = 0; i < stardustCount * 3; i++) stardustPos[i] = (Math.random() - 0.5) * 800;
    const stardustGeo = new THREE.BufferGeometry();
    stardustGeo.setAttribute('position', new THREE.BufferAttribute(stardustPos, 3));
    const stardustMat = new THREE.PointsMaterial({ color: 0x444455, size: 0.3, transparent: true, opacity: 0.4 });
    stardust = new THREE.Points(stardustGeo, stardustMat);
    spaceGroup.add(stardust);

    // Post-procesado Bloom
    const renderPass = new RenderPass(scene, camera);
    const bloomPass  = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 1.5, 0.4, 0.85);
    bloomPass.threshold = 0.5;
    bloomPass.strength  = 0.7;
    bloomPass.radius    = 0.15;
    composer = new EffectComposer(renderer);
    composer.addPass(renderPass);
    composer.addPass(bloomPass);

    // Controles orbitales
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance   = 15;
    controls.maxDistance   = 700;
    controls.addEventListener('start', () => { state.autoRotate = false; });

    // Retícula de selección
    const reticleGeo = new THREE.RingGeometry(1.5, 1.8, 32);
    const reticleMat = new THREE.MeshBasicMaterial({ color: 0x00d4ff, transparent: true, opacity: 0, side: THREE.DoubleSide });
    reticle = new THREE.Mesh(reticleGeo, reticleMat);
    scene.add(reticle);

    // Responsive
    window.addEventListener('resize', () => {
        camera.aspect = innerWidth / innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(innerWidth, innerHeight);
        composer.setSize(innerWidth, innerHeight);
    });
}

export function cleanupWebGL() {
    state.disposables.forEach(obj => { try { obj.dispose(); } catch (_) {} });
    state.disposables.length = 0;
    spaceGroup?.clear();
}

export function cleanupAll() {
    try { state.cameraUtilsRef?.stop();  } catch (_) {}
    try { state.handsRef?.close();       } catch (_) {}
    cleanupWebGL();
}
