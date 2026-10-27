export interface CameraSettings {
  deviceId: string;
  width: number;
  height: number;
  fps: number;
}

export interface GlobalSettings {
  smoothing: number;
  expression: number;
  headSensitivity: number;
  mirror: boolean;
  camera: CameraSettings;
}

export type ModelMode = 'auto' | 'avatar' | 'rigid';

export interface ModelSettings {
  mode: ModelMode;
  px: number;
  py: number;
  pz: number;
  scale: number;
  rx: number;
  ry: number;
  rz: number;
  flipX: boolean;
  flipY: boolean;
  flipZ: boolean;
  neutral?: { q: number[]; p: number[] };
}

export const DEFAULT_GLOBAL: GlobalSettings = {
  smoothing: 0.5,
  expression: 1,
  headSensitivity: 1,
  mirror: true,
  camera: { deviceId: '', width: 1280, height: 720, fps: 30 },
};

export const DEFAULT_MODEL_SETTINGS: ModelSettings = {
  mode: 'auto',
  px: 0,
  py: 0,
  pz: 0,
  scale: 1,
  rx: 0,
  ry: 0,
  rz: 0,
  flipX: false,
  flipY: false,
  flipZ: false,
};

const KEY = 'glboverlay.settings';
const LAST_KEY = 'glboverlay.lastModel';

export function loadGlobal(): GlobalSettings {
  const base = structuredClone(DEFAULT_GLOBAL);
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const p = JSON.parse(raw) as Partial<GlobalSettings>;
      Object.assign(base, p, { camera: { ...base.camera, ...(p.camera ?? {}) } });
    }
  } catch {
    /* ignore corrupt or unavailable storage */
  }
  return base;
}

export function saveGlobal(s: GlobalSettings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* ignore */
  }
}

export function getLastModelId(): string | null {
  try {
    return localStorage.getItem(LAST_KEY);
  } catch {
    return null;
  }
}

export function setLastModelId(id: string | null): void {
  try {
    if (id) localStorage.setItem(LAST_KEY, id);
    else localStorage.removeItem(LAST_KEY);
  } catch {
    /* ignore */
  }
}
