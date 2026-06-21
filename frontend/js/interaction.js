import * as THREE from 'three';
import { state } from './state.js';
import { camera, spaceGroup, stardust, composer, controls, reticle } from './scene.js';

/*
Interacción: raycasting para hover/selección de estrellas, bucle de animación y listeners de mouse.
*/

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 1.2;

// Objetos THREE cacheados — evita allocations y GC pressure a 60fps
const _quatY = new THREE.Quaternion();
const _quatX = new THREE.Quaternion();
const _axisY = new THREE.Vector3(0, 1, 0);
const _axisX = new THREE.Vector3(1, 0, 0);

const handCursor = document.getElementById('hand-cursor');
const tooltip    = document.getElementById('tooltip');

// ==========================================
// SELECCIÓN DE ESTRELLA (click / pellizco)
// ==========================================
export function seleccionarEstrella() {
    if (state.hoveredStarIndex === null) {
        document.getElementById('target-panel').style.display = 'none';
        document.getElementById('info-panel').style.display   = 'none';
        return;
    }

    const idx   = state.hoveredStarIndex;
    const nombre = state.catalog.names[idx]  || 'Desconocido';
    const teff   = Math.round(state.catalog.temps[idx]  || 0);
    const radio  = (state.catalog.radii[idx] || 0).toFixed(2);
    const tc     = state.catalog.targets[idx];

    document.getElementById('target-panel').style.display = 'block';
    document.getElementById('t-selected-name').textContent = nombre;

    const typeText  = tc === 3 ? '♦ Hallazgo IA'
                    : tc === 2 ? '★ Tierra 2.0 (Habitable)'
                    : tc === 1 ? 'Mundo Exótico'
                    : tc === 0 ? 'Mundo Inhóspito'
                    : 'Desconocido';
    const typeColor = tc === 3 ? 'var(--cyan)'
                    : tc === 2 ? 'var(--gold)'
                    : tc === 1 ? '#e67e22'
                    : 'rgba(255,255,255,0.5)';

    let desc = `Este sistema exoplanetario orbita una estrella con temperatura superficial de ${teff} K. `;
    if      (tc === 3) desc += `Ha sido etiquetado por nuestra IA por su alto IHP. Requiere confirmación telemétrica pero presenta características atmosféricas prometedoras.`;
    else if (tc === 2) desc += `Con un radio de ${radio} R⊕ y en la zona de habitabilidad, es un fuerte candidato para albergar agua líquida y condiciones favorables para la vida.`;
    else if (tc === 1) desc += `Sus características anómalas y su tamaño de ${radio} R⊕ lo categorizan como un mundo exótico, sugiriendo composición química inusual o densas capas atmosféricas.`;
    else               desc += `Debido a la radiación estelar y su radio de ${radio} R⊕, las condiciones son extremadamente hostiles. Probablemente un gigante gaseoso abrasador o roca estéril.`;

    const ipType = document.getElementById('ip-type');
    ipType.textContent = typeText;
    ipType.style.color = typeColor;
    document.getElementById('ip-name').textContent  = nombre;
    document.getElementById('ip-teff').textContent  = `TEMP ESTELAR: ${teff} K`;
    document.getElementById('ip-rade').textContent  = `RADIO PLANETARIO: ${radio} R⊕`;
    document.getElementById('ip-desc').textContent  = desc;

    const infoPanel = document.getElementById('info-panel');
    infoPanel.style.borderLeftColor = typeColor;
    infoPanel.style.display         = 'flex';
}

// ==========================================
// HOVER: RAYCASTING Y TOOLTIP
// ==========================================
export function renderHover() {
    const targets = [];
    if (state.pointsRef)      targets.push(state.pointsRef);
    if (state.grialPointsRef) targets.push(state.grialPointsRef);
    if (state.iaPointsRef)    targets.push(state.iaPointsRef);

    if (targets.length === 0 || state.virtualPointer.x === -999) {
        tooltip.style.display    = 'none';
        state.hoveredStarIndex   = null;
        reticle.material.opacity = 0;
        return;
    }

    raycaster.setFromCamera(state.virtualPointer, camera);
    const intersects = raycaster.intersectObjects(targets);

    if (intersects.length > 0) {
        const hit = intersects[0];
        state.hoveredStarIndex   = hit.object.userData.indexMap[hit.index];
        const idx = state.hoveredStarIndex;
        const tc  = state.catalog.targets[idx];

        document.getElementById('tt-name').textContent = state.catalog.names[idx] || '—';
        document.getElementById('tt-teff').textContent = `Estrella: ${Math.round(state.catalog.temps[idx] || 0)} K`;
        document.getElementById('tt-rade').textContent = `Radio: ${(state.catalog.radii[idx] || 0).toFixed(2)} R⊕`;

        const ihpVal    = state.catalog.ihps?.[idx]         || 0;
        const iaVal     = state.catalog.score_ia?.[idx]     || 0;
        const hellerVal = state.catalog.score_heller?.[idx] || 0;

        const ttIhp = document.getElementById('tt-ihp');
        if (tc === 3 || tc === 2) {
            ttIhp.style.display = 'block';
            ttIhp.innerHTML = `
                <div style="color:var(--cyan)">IHP TOTAL: ${ihpVal}%</div>
                <div style="font-size:8px; opacity:0.7; margin-top:2px; color:#fff;">IA: ${iaVal}% | Heller: ${hellerVal}%</div>
            `;
        } else {
            ttIhp.style.display = 'none';
        }

        const grialText = document.getElementById('tt-grial');
        grialText.textContent = tc === 3 ? '♦ DESCUBRIMIENTO IA'
                              : tc === 2 ? '★ TIERRA 2.0 (GRIAL)'
                              : tc === 1 ? 'Mundo Exótico'
                              : tc === 0 ? 'Mundo Inhóspito'
                              : 'Desconocido';
        grialText.style.color = tc === 3 ? 'var(--cyan)' : tc === 2 ? 'var(--gold)' : 'rgba(255,255,255,0.4)';

        tooltip.style.display    = 'block';
        document.body.style.cursor = 'pointer';
    } else {
        reticle.material.opacity   = 0;
        tooltip.style.display      = 'none';
        document.body.style.cursor = 'crosshair';
        state.hoveredStarIndex     = null;
    }
}

// ==========================================
// BUCLE DE ANIMACIÓN PRINCIPAL
// ==========================================
export function animate() {
    state.animFrameId = requestAnimationFrame(animate);

    if (state.autoRotate) {
        state.autoRotateY     += 0.00050;
        spaceGroup.rotation.y  = state.autoRotateY;
    } else if (state.isGrabbingSpace) {
        spaceGroup.quaternion.slerp(state.targetQuaternion, 0.04);
    } else if (state.rotVelX !== 0 || state.rotVelY !== 0) {
        // Inercia post-agarre: el target sigue avanzando con velocidad que decae
        _quatY.setFromAxisAngle(_axisY, state.rotVelY);
        _quatX.setFromAxisAngle(_axisX, state.rotVelX);
        state.targetQuaternion.premultiply(_quatY).premultiply(_quatX);
        state.rotVelY *= 0.88;
        state.rotVelX *= 0.88;
        if (Math.abs(state.rotVelY) < 0.0001) state.rotVelY = 0;
        if (Math.abs(state.rotVelX) < 0.0001) state.rotVelX = 0;
        spaceGroup.quaternion.slerp(state.targetQuaternion, 0.04);
    }

    if (stardust) {
        stardust.rotation.y -= 0.001;
        stardust.rotation.x += 0.002;
    }

    renderHover();
    controls.update();
    composer.render();
}

// ==========================================
// EVENT LISTENERS DE MOUSE
// ==========================================
export function initInteraction() {
    window.addEventListener('mousemove', (e) => {
        if (!handCursor.classList.contains('pointing')) {
            state.virtualPointer.x = (e.clientX / innerWidth)  *  2 - 1;
            state.virtualPointer.y = (e.clientY / innerHeight)  * -2 + 1;
            tooltip.style.left = `${e.clientX + 24}px`;
            tooltip.style.top  = `${e.clientY - 16}px`;
        }
    });

    window.addEventListener('click', () => {
        if (state.virtualPointer.x !== -999) seleccionarEstrella();
    });
}
