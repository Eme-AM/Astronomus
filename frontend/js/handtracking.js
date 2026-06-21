import * as THREE from 'three';
import { state } from './state.js';
import { camera, controls, spaceGroup } from './scene.js';
import { seleccionarEstrella } from './interaction.js';

/*
Hand tracking: inicializa MediaPipe Hands, 
detecta gestos (rotar, apuntar, zoom, clic) y 
mapea la mano a la escena.
*/

const handCursor = document.getElementById('hand-cursor');
const tooltip    = document.getElementById('tooltip');

// Objetos THREE cacheados — evita allocations y GC pressure a cada frame de cámara
const _quatY = new THREE.Quaternion();
const _quatX = new THREE.Quaternion();
const _axisY = new THREE.Vector3(0, 1, 0);
const _axisX = new THREE.Vector3(1, 0, 0);

export function initWebcamAndAI() {
    const videoElement = document.getElementById('webcam');
    const statusText   = document.getElementById('ai-status');

    if (!window.Hands || !window.Camera) {
        statusText.textContent = 'Error de IA';
        return;
    }

    state.handsRef = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    const hands = state.handsRef;
    hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.6, minTrackingConfidence: 0.8 });

    let handsDetected = false;
    let prevGrabX = null;
    let prevGrabY = null;
    let idleTimer  = null;

    hands.onResults((results) => {
        handsDetected = !!results.multiHandLandmarks?.length;

        state.isGrabbingSpace = false;
        handCursor.style.display = 'none';
        state.virtualPointer.x   = -999;

        if (!handsDetected) {
            state.lastZoomDist = 0; state.wasPinching = false;
            statusText.style.color = 'gray'; statusText.textContent = 'Buscando manos...';
            if (!idleTimer) {
                idleTimer = setTimeout(() => {
                    state.autoRotateY = spaceGroup.rotation.y;
                    state.autoRotate  = true;
                    idleTimer = null;
                }, 3000);
            }
            return;
        }

        if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
        state.autoRotate = false;

        const parsedHands = results.multiHandLandmarks.filter(hand => hand?.length >= 21).map(hand => {
            const wrist      = hand[0];
            const thumbTip   = hand[4],  thumbIP    = hand[3];
            const indexPip   = hand[6],  indexTip   = hand[8];
            const middlePip  = hand[10], middleTip  = hand[12], middleBase = hand[9];
            const ringPip    = hand[14], ringTip    = hand[16];
            const pinkyPip   = hand[18], pinkyTip   = hand[20];

            const palmCenterX = (wrist.x + middleBase.x) * 0.5;
            const palmCenterY = (wrist.y + middleBase.y) * 0.5;

            const thumbCurled  = Math.hypot(thumbTip.x  - palmCenterX, thumbTip.y  - palmCenterY) < Math.hypot(thumbIP.x  - palmCenterX, thumbIP.y  - palmCenterY) * 1.1;
            const indexCurled  = Math.hypot(indexTip.x  - wrist.x, indexTip.y  - wrist.y) < Math.hypot(indexPip.x  - wrist.x, indexPip.y  - wrist.y) * 1.08;
            const middleCurled = Math.hypot(middleTip.x - wrist.x, middleTip.y - wrist.y) < Math.hypot(middlePip.x - wrist.x, middlePip.y - wrist.y) * 1.08;
            const ringCurled   = Math.hypot(ringTip.x   - wrist.x, ringTip.y   - wrist.y) < Math.hypot(ringPip.x   - wrist.x, ringPip.y   - wrist.y) * 1.08;
            const pinkyCurled  = Math.hypot(pinkyTip.x  - wrist.x, pinkyTip.y  - wrist.y) < Math.hypot(pinkyPip.x  - wrist.x, pinkyPip.y  - wrist.y) * 1.12;

            const curledCount = [indexCurled, middleCurled, ringCurled, pinkyCurled].filter(Boolean).length;
            const palmSize    = Math.hypot(wrist.x - middleBase.x, wrist.y - middleBase.y);
            const pinchDist   = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
            const indexToPalm = Math.hypot(indexTip.x - middleBase.x, indexTip.y - middleBase.y);

            const isPinching       = (pinchDist < palmSize * 0.4) && (indexToPalm > palmSize * 0.6);
            const middlePinchDist  = Math.hypot(thumbTip.x - middleTip.x, thumbTip.y - middleTip.y);
            const isMiddlePinching = middlePinchDist < palmSize * 0.25;
            const isPointing = !indexCurled && middleCurled && ringCurled && !isPinching;
            const isGrabbing = indexCurled && middleCurled && ringCurled && !isPinching;

            return { hand, wrist, indexTip, middleBase, isGrabbing, isPointing, isPinching, isMiddlePinching, isOpen: !thumbCurled && curledCount === 0 };
        });

        // Zoom con dos manos abiertas
        if (parsedHands.length === 2 && parsedHands[0].isOpen && parsedHands[1].isOpen) {
            statusText.style.color = 'var(--gold)'; statusText.textContent = 'Escalando (Zoom)';
            const dist = Math.hypot(parsedHands[0].wrist.x - parsedHands[1].wrist.x, parsedHands[0].wrist.y - parsedHands[1].wrist.y);
            if (state.lastZoomDist > 0) {
                const delta = dist - state.lastZoomDist;
                let radius  = camera.position.length();
                radius = Math.max(controls.minDistance, Math.min(controls.maxDistance, radius - delta * 250));
                camera.position.setLength(radius);
            }
            state.lastZoomDist = dist; state.wasPinching = false;
            return;
        }

        state.lastZoomDist = 0;

        const pointer = parsedHands.find(h => h.isPointing);
        const pincher = parsedHands.find(h => h.isPinching);
        const grabber = parsedHands.find(h => h.isGrabbing);

        // Rotación relativa del espacio con puño cerrado.
        // Se acumula el DELTA de posición (no la posición absoluta) para que
        // el ángulo de la escena dependa de cuánto se movió la mano, no de dónde está.
        if (grabber) {
            state.isGrabbingSpace = true;
            const h = grabber;
            const handCenterX = (h.wrist.x + h.middleBase.x + h.hand[5].x + h.hand[17].x) * 0.25;
            const handCenterY = (h.wrist.y + h.middleBase.y + h.hand[5].y + h.hand[17].y) * 0.25;

            if (prevGrabX !== null) {
                // Clamp para absorber glitches de tracking (saltos bruscos)
                const dx = Math.max(-0.08, Math.min(0.08, handCenterX - prevGrabX));
                const dy = Math.max(-0.08, Math.min(0.08, handCenterY - prevGrabY));
                // premultiply = ejes del MUNDO (no locales) → sin gimbal lock.
                // La cámara es espejada: dx<0 cuando el usuario mueve a la derecha,
                // por eso angleY=dx (sin negar) produce rotación Y negativa = gira a la derecha.
                const angleY = -dx * Math.PI * 2;
                const angleX = dy * Math.PI * 2;
                _quatY.setFromAxisAngle(_axisY, angleY);
                _quatX.setFromAxisAngle(_axisX, angleX);
                state.targetQuaternion.premultiply(_quatY).premultiply(_quatX);
                state.rotVelY = angleY;
                state.rotVelX = angleX;
            } else {
                // Primer frame del agarre: sincronizar con posición visual actual y limpiar inercia
                state.targetQuaternion.copy(spaceGroup.quaternion);
                state.rotVelX = 0;
                state.rotVelY = 0;
            }
            prevGrabX = handCenterX;
            prevGrabY = handCenterY;
        } else {
            prevGrabX = null;
            prevGrabY = null;
            // No se resetea — la inercia en animate() desacelera suavemente
        }

        // Apuntado con dedo índice extendido
        if (pointer) {
            const screenX = (1 - pointer.indexTip.x) * window.innerWidth;
            const screenY =      pointer.indexTip.y  * window.innerHeight;
            state.smoothTargetX += (screenX - state.smoothTargetX) * 0.15;
            state.smoothTargetY += (screenY - state.smoothTargetY) * 0.15;

            handCursor.style.display = 'block';
            handCursor.style.left    = `${state.smoothTargetX}px`;
            handCursor.style.top     = `${state.smoothTargetY}px`;
            state.virtualPointer.x   =  (state.smoothTargetX / window.innerWidth)  *  2 - 1;
            state.virtualPointer.y   = -(state.smoothTargetY / window.innerHeight)  *  2 + 1;
            tooltip.style.left       = `${state.smoothTargetX + 24}px`;
            tooltip.style.top        = `${state.smoothTargetY - 16}px`;

            const singleHandClick = pointer.isMiddlePinching && parsedHands.length === 1;
            if (pincher || singleHandClick) {
                handCursor.className = 'pinching';
                if (!state.wasPinching) seleccionarEstrella();
                state.wasPinching = true;
            } else {
                handCursor.className  = 'pointing';
                state.wasPinching     = false;
            }
        } else {
            state.wasPinching = false;
        }

        // Feedback visual
        if      (pointer && pointer.isMiddlePinching && parsedHands.length === 1) { statusText.style.color = 'var(--gold)'; statusText.textContent = 'Clic (1 Mano)'; }
        else if (pointer && pincher)                                              { statusText.style.color = 'var(--gold)'; statusText.textContent = 'Clic (2 Manos)'; }
        else if (grabber && pointer)                  { statusText.style.color = 'var(--cyan)'; statusText.textContent = 'Rotando y Apuntando'; }
        else if (grabber)                             { statusText.style.color = 'white';       statusText.textContent = 'Rotando espacio'; }
        else if (pointer)                             { statusText.style.color = 'var(--cyan)'; statusText.textContent = 'Apuntando...'; }
        else if (pincher)                             { statusText.style.color = 'gray';        statusText.textContent = 'Falta mano para apuntar'; }
        else                                          { statusText.style.color = 'gray';        statusText.textContent = parsedHands.length === 2 ? '2 Manos detectadas' : '1 Mano detectada'; }
    });

    // Frame skipping adaptivo: procesa 1 de cada IDLE_SKIP frames cuando no hay manos,
    // vuelve a full rate en cuanto se detecta una (handsDetected se actualiza en onResults).
    const IDLE_SKIP = 3;
    let frameCount = 0;

    state.cameraUtilsRef = new Camera(videoElement, {
        onFrame: async () => {
            frameCount++;
            if (handsDetected || frameCount % IDLE_SKIP === 0) {
                await hands.send({ image: videoElement });
            }
        },
        width: 320, height: 240,
    });
    state.cameraUtilsRef.start().then(() => { statusText.textContent = 'Sistema IA Listo'; });
}
