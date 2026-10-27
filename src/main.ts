import './style.css';
import { MAX_BYTES, ModelError, ModelLoader, WARN_BYTES, pickModelFiles, totalSize, type LoadedModel } from './loader';
import { ModelRig } from './rig';
import {
  DEFAULT_GLOBAL,
  DEFAULT_MODEL_SETTINGS,
  getLastModelId,
  loadGlobal,
  saveGlobal,
  setLastModelId,
  type GlobalSettings,
} from './settings';
import { Stage } from './stage';
import {
  ModelStore,
  StorageFullError,
  formatBytes,
  requestPersistence,
  storageEstimate,
  type ModelMeta,
  type StoredFile,
} from './storage';
import { Tracker, TrackerError } from './tracker';
import { UI } from './ui';

const canvas = document.getElementById('view') as HTMLCanvasElement;
const video = document.getElementById('cam') as HTMLVideoElement;
const uiRoot = document.getElementById('ui') as HTMLElement;

const global: GlobalSettings = loadGlobal();
let models: ModelMeta[] = [];
let active: ModelMeta | null = null;
let rig: ModelRig | null = null;
let loadGen = 0;
let green = false;

const store = new ModelStore();
const tracker = new Tracker(video);
tracker.processor.setSmoothing(global.smoothing);

const ui = new UI(
  uiRoot,
  {
    addFiles: (f) => void addFiles(f),
    selectModel: (id) => void selectModel(id),
    renameModel: (id, name) => void renameModel(id, name),
    deleteModel: (id) => void deleteModel(id),
    globalChanged: (restart) => {
      saveGlobal(global);
      tracker.processor.setSmoothing(global.smoothing);
      if (restart) void startTracking();
    },
    modelSettingsChanged: () => {
      if (!active) return;
      stage?.applyCalibration(active.settings);
      applyModeChange();
      persistMeta();
    },
    resetGlobal: () => {
      Object.assign(global, structuredClone(DEFAULT_GLOBAL));
      saveGlobal(global);
      tracker.processor.setSmoothing(global.smoothing);
      void startTracking();
    },
    resetModel: () => {
      if (!active) return;
      const neutral = active.settings.neutral;
      active.settings = { ...DEFAULT_MODEL_SETTINGS, neutral };
      stage?.applyCalibration(active.settings);
      applyModeChange();
      persistMeta();
    },
    recalibrate,
    retryCamera: () => void startTracking(),
  },
  () => global,
  () => active?.settings ?? null,
);

let stage: Stage | null = null;
let loader: ModelLoader | null = null;

function isChrome(): boolean {
  const uad = (navigator as Navigator & { userAgentData?: { brands: { brand: string }[] } }).userAgentData;
  if (uad) return uad.brands.some((b) => b.brand === 'Google Chrome' || b.brand === 'Chromium');
  return /Chrome\//.test(navigator.userAgent) && !/Edg\/|OPR\//.test(navigator.userAgent);
}

// ---------------------------------------------------------------- startup

async function init(): Promise<void> {
  try {
    stage = new Stage(canvas);
    loader = new ModelLoader(stage.renderer);
  } catch (e) {
    ui.fatal(`WebGL is unavailable or the GPU could not be initialised (${(e as Error).message}). Enable hardware acceleration in chrome://settings/system.`);
    return;
  }
  window.addEventListener('resize', () => stage?.resize());
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    ui.showError('The GPU context was lost. Reload the page to continue.', { label: 'Reload', run: () => location.reload() });
  });

  if (!isChrome()) ui.showBanner('glboverlay is built for desktop Chrome. Other browsers may not work correctly.');
  ui.showToast('Press I for help');

  tracker.onStatus = (s) => ui.setTrackerStatus(s);
  tracker.onDevices = () => void refreshDevices();
  tracker.onError = (e) => showTrackerError(e);
  void refreshDevices();
  void requestWakeLock();

  try {
    models = await store.list();
  } catch (e) {
    ui.showError(`Browser storage is unavailable, so models cannot be saved (${(e as Error).message}).`);
    models = [];
  }
  ui.setModels(models, null);
  void refreshStorage();

  const last = getLastModelId();
  const target = models.find((m) => m.id === last) ?? models[0];
  if (target) await selectModel(target.id);
  else ui.showDropzone(true);

  requestAnimationFrame(frame);
}

async function requestWakeLock(): Promise<void> {
  try {
    await (navigator as Navigator & { wakeLock?: { request(t: string): Promise<unknown> } }).wakeLock?.request('screen');
  } catch {
    /* optional */
  }
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') void requestWakeLock();
  });
}

async function refreshDevices(): Promise<void> {
  ui.setDevices(await tracker.listDevices());
}

async function refreshStorage(): Promise<void> {
  let persisted = false;
  try {
    persisted = (await navigator.storage?.persisted?.()) ?? false;
  } catch {
    /* ignore */
  }
  ui.setStorage(await storageEstimate(), persisted);
}

// ---------------------------------------------------------------- tracking

async function startTracking(): Promise<void> {
  ui.clearError();
  try {
    await tracker.start(global.camera);
  } catch (e) {
    showTrackerError(e instanceof TrackerError ? e : new TrackerError('other', (e as Error).message));
  }
}

function showTrackerError(e: TrackerError): void {
  ui.showError(e.message, { label: 'Retry', run: () => void startTracking() });
}

function recalibrate(): void {
  if (!rig || !active) return;
  if (!rig.recalibrate(tracker.processor.state)) ui.showError('No face detected yet, so there is nothing to calibrate.');
}

// ---------------------------------------------------------------- models

function persistMeta(): void {
  if (!active) return;
  const m = active;
  window.clearTimeout(persistTimer);
  persistTimer = window.setTimeout(() => {
    store.updateMeta(m).catch((e) => ui.showError(e instanceof StorageFullError ? 'Storage is full: settings were not saved.' : `Could not save settings: ${(e as Error).message}`));
  }, 400);
}
let persistTimer = 0;

function describeMode(): string {
  if (!rig) return '';
  const i = rig.info;
  const bits = [`Detected: ${i.detectedMode}`, i.vrm ? 'VRM' : '', i.perfectSync ? 'Perfect Sync' : '', i.morphTargets ? `${i.morphTargets} ARKit morphs` : 'no ARKit morphs'];
  if (rig.mode === 'avatar' && i.detectedMode === 'rigid') bits.push('no head bone found, so nothing will move');
  return bits.filter(Boolean).join(' · ');
}

/** (Re)build the rig for the model currently in the stage. */
function buildRig(model: LoadedModel, meta: ModelMeta): void {
  rig?.dispose();
  const r = new ModelRig(model, stage!.scene, meta.settings.mode, meta.settings.neutral);
  r.onNeutralChange = (n) => {
    meta.settings.neutral = n;
    persistMeta();
  };
  rig = r;
  stage!.refit(r.mode === 'avatar', () => r.headWorldY());
  stage!.applyCalibration(meta.settings);
  ui.setModeNote(describeMode());
}

let currentModel: LoadedModel | null = null;

function applyModeChange(): void {
  if (!currentModel || !active || !rig) return;
  if (active.settings.mode === 'auto' ? rig.info.detectedMode === rig.mode : active.settings.mode === rig.mode) {
    ui.setModeNote(describeMode());
    return;
  }
  // Mode switched: reset any bone pose and refit.
  buildRig(currentModel, active);
}

async function presentModel(meta: ModelMeta, files: StoredFile[], fresh: boolean): Promise<void> {
  const gen = ++loadGen;
  ui.setLoading(`Loading ${meta.name}…`);
  ui.clearError();
  let model: LoadedModel;
  try {
    model = await loader!.load(files, meta.main);
  } finally {
    if (gen === loadGen) ui.setLoading(null);
  }
  if (gen !== loadGen) {
    model.dispose();
    throw new Error('superseded');
  }
  rig?.dispose();
  rig = null;
  stage!.setModel(model);
  currentModel = model;
  active = meta;
  buildRig(model, meta);
  ui.showDropzone(false);
  setLastModelId(meta.id);
  ui.setModels(models, meta.id);
  if (fresh) {
    // First render at rest pose for the thumbnail.
    rig!.update(0, 0, tracker.processor.state, global, stage!.pose);
    const thumb = await stage!.thumbnail();
    if (thumb) meta.thumbnail = thumb;
  }
  if (!tracker.isRunning) void startTracking();
}

async function selectModel(id: string): Promise<void> {
  const meta = models.find((m) => m.id === id);
  if (!meta) return;
  try {
    const files = await store.getFiles(id);
    await presentModel(meta, files, false);
  } catch (e) {
    if ((e as Error).message === 'superseded') return;
    ui.showError(e instanceof ModelError ? e.message : `Could not open "${meta.name}": ${(e as Error).message}`);
    if (!active) ui.showDropzone(models.length === 0);
  }
}

async function addFiles(input: File[]): Promise<void> {
  if (!stage || !loader) return;
  let picked;
  try {
    picked = pickModelFiles(input);
  } catch (e) {
    ui.showError((e as Error).message);
    return;
  }
  const size = totalSize(picked.files);
  if (size > MAX_BYTES) {
    ui.showError(
      `This model is ${formatBytes(size)}, over the ${formatBytes(MAX_BYTES)} limit. Compress it first (Draco or Meshopt geometry compression, smaller textures) with gltf-transform or Blender.`,
    );
    return;
  }
  if (size > WARN_BYTES) {
    const ok = await ui.confirm(
      `This model is ${formatBytes(size)}. Large models load slowly and may stutter. Compressing with Draco or Meshopt is recommended. Add it anyway?`,
      'Add anyway',
    );
    if (!ok) return;
  }

  const baseName = picked.main.name.replace(/\.[^.]+$/, '');
  const meta: ModelMeta = {
    id: crypto.randomUUID(),
    name: baseName,
    main: picked.main.name,
    size,
    date: Date.now(),
    settings: structuredClone(DEFAULT_MODEL_SETTINGS),
  };
  try {
    await presentModel(meta, picked.files, true);
  } catch (e) {
    if ((e as Error).message === 'superseded') return;
    ui.showError(e instanceof ModelError ? e.message : `Could not load model: ${(e as Error).message}`);
    return;
  }

  void requestPersistence().then(refreshStorage);
  try {
    await store.add(meta, picked.files);
    models = [meta, ...models];
    ui.setModels(models, meta.id);
    void refreshStorage();
  } catch (e) {
    ui.showError(
      e instanceof StorageFullError
        ? 'Browser storage is full, so this model was not saved. It will work for this session. Delete models you no longer need in the model manager (M).'
        : `The model loaded but could not be saved: ${(e as Error).message}`,
    );
  }
}

async function renameModel(id: string, name: string): Promise<void> {
  const m = models.find((x) => x.id === id);
  if (!m) return;
  m.name = name;
  try {
    await store.updateMeta(m);
  } catch (e) {
    ui.showError(`Could not rename: ${(e as Error).message}`);
  }
  ui.setModels(models, active?.id ?? null);
}

async function deleteModel(id: string): Promise<void> {
  const m = models.find((x) => x.id === id);
  if (!m) return;
  if (!(await ui.confirm(`Delete "${m.name}"? This cannot be undone.`, 'Delete', true))) return;
  try {
    await store.remove(id);
  } catch (e) {
    ui.showError(`Could not delete: ${(e as Error).message}`);
    return;
  }
  models = models.filter((x) => x.id !== id);
  if (active?.id === id) {
    loadGen++;
    rig?.dispose();
    rig = null;
    stage?.clearModel();
    currentModel = null;
    active = null;
    setLastModelId(null);
    if (models.length) {
      ui.setModels(models, null);
      await selectModel(models[0].id);
    } else {
      tracker.stop();
      ui.closePanels();
      ui.showDropzone(true);
    }
  }
  ui.setModels(models, active?.id ?? null);
  void refreshStorage();
}

// ---------------------------------------------------------------- render loop

let lastFrame = performance.now();

function frame(): void {
  requestAnimationFrame(frame);
  const nowMs = performance.now();
  const dt = Math.min((nowMs - lastFrame) / 1000, 0.1);
  lastFrame = nowMs;
  if (rig && stage && active) {
    rig.update(dt, performance.now() / 1000, tracker.processor.state, global, stage.pose);
  }
  stage?.render();
}

// ---------------------------------------------------------------- keyboard

function setGreen(on: boolean): void {
  green = on;
  document.documentElement.classList.toggle('green', on);
  if (on) {
    ui.closePanels();
    (document.activeElement as HTMLElement | null)?.blur();
  }
}

window.addEventListener('keydown', (e) => {
  const t = e.target as HTMLElement | null;
  const typing = !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
  if (e.key === 'Escape') {
    if (ui.closePanels()) e.preventDefault();
    return;
  }
  if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
  switch (e.key.toLowerCase()) {
    case 'c':
      setGreen(!green);
      break;
    case 'i':
      if (!green) ui.toggle(ui.info);
      break;
    case 'm':
      if (!green) {
        ui.toggle(ui.models);
        void refreshStorage();
      }
      break;
    case 's':
      if (!green) ui.toggle(ui.settings);
      break;
    case 'r':
      recalibrate();
      break;
    default:
      return;
  }
  e.preventDefault();
});

if (import.meta.env.DEV) {
  Object.assign(window, { __glb: { tracker, global, getRig: () => rig, getStage: () => stage } });
}

void init();
