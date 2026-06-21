import * as THREE from 'three';
import { state } from './state.js';
import { spaceGroup } from './scene.js';

/**
 * Catálogo: consume la API de exoplanetas, construye la 
 * geometría Three.js para las 4 clases y actualiza el HUD
 */

const API_URL = window.location.protocol === 'file:'
    ? 'http://localhost:8000/api/exoplanets'
    : '/api/exoplanets';

export function tempToRGB(teff) {
    const t = Math.max(2500, Math.min(10000, teff || 5778));
    let r, g, b;
    if      (t < 3700) { r = 1; g = 0.3 + (t - 2500) / 4167; b = 0; }
    else if (t < 5200) { r = 1; g = 0.6 + (t - 3700) / 7500; b = 0.05; }
    else if (t < 6000) { r = 1; g = 0.9 + (t - 5200) / 8000; b = 0.4 + (t - 5200) / 2000; }
    else if (t < 7500) { r = 1; g = 1;                         b = 0.65 + (t - 6000) / 4286; }
    else               { r = 0.75 + (10000 - t) / 10000; g = 0.88; b = 1; }
    return [Math.min(1, r), Math.min(1, g), Math.min(1, b)];
}

export function createStarTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 64; canvas.height = 64;
    const ctx = canvas.getContext('2d');
    const cx = 32, cy = 32, spikes = 5, outerRadius = 30, innerRadius = 12;
    let rot = Math.PI / 2 * 3;
    let x = cx, y = cy;
    const step = Math.PI / spikes;
    ctx.beginPath();
    ctx.moveTo(cx, cy - outerRadius);
    for (let i = 0; i < spikes; i++) {
        x = cx + Math.cos(rot) * outerRadius; y = cy + Math.sin(rot) * outerRadius; ctx.lineTo(x, y); rot += step;
        x = cx + Math.cos(rot) * innerRadius; y = cy + Math.sin(rot) * innerRadius; ctx.lineTo(x, y); rot += step;
    }
    ctx.lineTo(cx, cy - outerRadius); ctx.closePath();
    const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, outerRadius);
    gradient.addColorStop(0,   'rgba(255, 255, 255, 1)');
    gradient.addColorStop(0.3, 'rgba(255, 255, 255, 0.9)');
    gradient.addColorStop(1,   'rgba(255, 255, 255, 0)');
    ctx.fillStyle = gradient; ctx.fill();
    return new THREE.CanvasTexture(canvas);
}

export async function loadCatalog() {
    const setStatus = (msg, pct) => {
        document.getElementById('status-msg').textContent = msg;
        document.getElementById('prog').style.width = pct + '%';
    };
    setStatus('Conectando con la API...', 10);

    // Limpiar geometrías de una carga anterior
    state.disposables.forEach(obj => { try { obj.dispose(); } catch (_) {} });
    state.disposables.length = 0;
    spaceGroup?.clear();

    const starTexture = createStarTexture();
    state.disposables.push(starTexture);

    const fetchController = new AbortController();
    const fetchTimeout = setTimeout(() => fetchController.abort(), 20000);

    try {
        let res;
        try {
            res = await fetch(API_URL, { signal: fetchController.signal });
            clearTimeout(fetchTimeout);
        } catch {
            clearTimeout(fetchTimeout);
            // Datos de demostración cuando la API no está disponible
            res = { ok: true, json: async () => ({
                meta: { total: 1000, labeled: 100, griales: 5, ia_candidates: 12, exoticos: 80, inhospitos: 903 },
                positions:     Array.from({ length: 3000 }, () => (Math.random() - 0.5) * 10000),
                temperatures:  Array.from({ length: 1000 }, () => Math.random() * 8000 + 2000),
                radii:         Array.from({ length: 1000 }, () => Math.random() * 5),
                target_classes: Array.from({ length: 1000 }, () => Math.random() > 0.95 ? 2 : 0),
                names:         Array.from({ length: 1000 }, (_, i) => `Kepler-${i}`),
                ihp:           [], score_ia: [], score_heller: [],
            })};
        }

        const data = await res.json();

        state.catalog = {
            names:        data.names,
            temps:        data.temperatures,
            radii:        data.radii,
            targets:      data.target_classes,
            ihps:         data.ihp,
            score_ia:     data.score_ia,
            score_heller: data.score_heller,
        };

        setStatus(`Procesando ${data.meta.total} cuerpos celestes...`, 70);
        await new Promise(r => setTimeout(r, 50));

        // Clasificar índices por tipo
        const normIdx = [], grialIdx = [], iaIdx = [];
        data.target_classes.forEach((tc, i) => {
            if      (tc === 2) grialIdx.push(i);
            else if (tc === 3) iaIdx.push(i);
            else               normIdx.push(i);
        });

        const jitter = () => (Math.random() - 0.5) * 3.5;

        // Planetas normales (clases 0 y 1)
        const normPos = new Float32Array(normIdx.length * 3);
        const normCol = new Float32Array(normIdx.length * 3);
        normIdx.forEach((src, dst) => {
            normPos[dst * 3]     = data.positions[src * 3]     + jitter();
            normPos[dst * 3 + 1] = data.positions[src * 3 + 1] + jitter();
            normPos[dst * 3 + 2] = data.positions[src * 3 + 2] + jitter();
            const [r, g, b] = tempToRGB(data.temperatures[src]);
            normCol[dst * 3] = r; normCol[dst * 3 + 1] = g; normCol[dst * 3 + 2] = b;
        });
        const normGeo = new THREE.BufferGeometry();
        normGeo.setAttribute('position', new THREE.BufferAttribute(normPos, 3));
        normGeo.setAttribute('color',    new THREE.BufferAttribute(normCol, 3));
        const normMat = new THREE.PointsMaterial({
            vertexColors: true, size: 2.5, sizeAttenuation: true,
            transparent: true, opacity: 0.50, map: starTexture,
            alphaTest: 0.05, depthWrite: false,
        });
        state.pointsRef = new THREE.Points(normGeo, normMat);
        state.pointsRef.userData.indexMap = normIdx;
        spaceGroup.add(state.pointsRef);
        state.disposables.push(normGeo, normMat);

        // Griales (clase 2)
        if (grialIdx.length > 0) {
            const gPos = new Float32Array(grialIdx.length * 3);
            grialIdx.forEach((src, dst) => {
                gPos[dst * 3] = data.positions[src * 3]; gPos[dst * 3 + 1] = data.positions[src * 3 + 1]; gPos[dst * 3 + 2] = data.positions[src * 3 + 2];
            });
            const gGeo  = new THREE.BufferGeometry(); gGeo.setAttribute('position', new THREE.BufferAttribute(gPos, 3));
            const gMat  = new THREE.PointsMaterial({ color: 0xf1c40f, size: 9.0, sizeAttenuation: true, transparent: true, map: starTexture, alphaTest: 0.05, depthWrite: false });
            const gGlow = new THREE.PointsMaterial({ color: 0xf39c12, size: 11.0, sizeAttenuation: true, transparent: true, opacity: 0.18 });
            state.grialPointsRef = new THREE.Points(gGeo, gMat);
            state.grialPointsRef.userData.indexMap = grialIdx;
            spaceGroup.add(state.grialPointsRef);
            spaceGroup.add(new THREE.Points(gGeo, gGlow));
            state.disposables.push(gGeo, gMat, gGlow);
        }

        // Hallazgos IA (clase 3)
        if (iaIdx.length > 0) {
            const iaPos = new Float32Array(iaIdx.length * 3);
            iaIdx.forEach((src, dst) => {
                iaPos[dst * 3] = data.positions[src * 3]; iaPos[dst * 3 + 1] = data.positions[src * 3 + 1]; iaPos[dst * 3 + 2] = data.positions[src * 3 + 2];
            });
            const iaGeo  = new THREE.BufferGeometry(); iaGeo.setAttribute('position', new THREE.BufferAttribute(iaPos, 3));
            const iaMat  = new THREE.PointsMaterial({ color: 0x00d4ff, size: 4.5, sizeAttenuation: true });
            const iaGlow = new THREE.PointsMaterial({ color: 0x00d4ff, size: 9.0, sizeAttenuation: true, transparent: true, opacity: 0.15 });
            state.iaPointsRef = new THREE.Points(iaGeo, iaMat);
            state.iaPointsRef.userData.indexMap = iaIdx;
            spaceGroup.add(state.iaPointsRef);
            spaceGroup.add(new THREE.Points(iaGeo, iaGlow));
            state.disposables.push(iaGeo, iaMat, iaGlow);
        }

        // Actualizar HUD
        document.getElementById('h-total').textContent       = data.meta.total.toLocaleString('es-AR') + ' EXOPLANETAS';
        document.getElementById('h-ia-candidates').textContent = (data.meta.ia_candidates || 0) + ' DETECTADOS';
        document.getElementById('h-griales').textContent     = (data.meta.griales   || 0) + ' CONFIRMADOS';
        document.getElementById('h-exoticos').textContent    = (data.meta.exoticos  || 0) + ' REGISTRADOS';
        document.getElementById('h-inhospitos').textContent  = (data.meta.inhospitos || 0) + ' DESCARTADOS';

        setStatus('Catálogo inicializado', 100);
        setTimeout(() => {
            document.getElementById('overlay').classList.add('fade-out');
            ['hud', 'legend', 'controls-hint', 'webcam-container'].forEach(id => document.getElementById(id).classList.add('show'));
        }, 500);

    } catch (error) {
        setStatus('ERROR CRÍTICO: No se pudo cargar el catálogo.', 0);
    }
}
