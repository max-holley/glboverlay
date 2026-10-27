import {
  ACESFilmicToneMapping,
  Box3,
  DirectionalLight,
  Euler,
  Group,
  HemisphereLight,
  PerspectiveCamera,
  PMREMGenerator,
  Scene,
  SRGBColorSpace,
  Vector3,
  WebGLRenderer,
} from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import type { LoadedModel } from './loader';
import type { ModelSettings } from './settings';

const FOV = 30;
/** Cap on rendered pixels: full-window MSAA at 2x DPR on a 4K screen costs far more than it shows in OBS. */
const MAX_PIXELS = 2560 * 1440;

function pixelRatio(w: number, h: number, quality: number): number {
  return Math.max(0.5, Math.min(quality, 2, Math.sqrt(MAX_PIXELS / (w * h))));
}

/** Renderer, lights, camera and the transform hierarchy that holds the current model. */
export class Stage {
  readonly renderer: WebGLRenderer;
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(FOV, 1, 0.05, 50);
  /** Rigid-mode head pose is applied here. */
  readonly pose = new Group();
  /** Per-model calibration offsets. */
  readonly calib = new Group();
  /** Auto-fit scale and centring. */
  readonly fit = new Group();

  private model: LoadedModel | null = null;
  private visibleH = 1.5;
  private targetY = 0;
  private box = new Box3();
  private v = new Vector3();
  private eul = new Euler();
  private quality = 1;

  constructor(readonly canvas: HTMLCanvasElement) {
    this.renderer = new WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      premultipliedAlpha: true,
      powerPreference: 'high-performance',
    });
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1;
    this.renderer.setClearColor(0x000000, 0);

    const pmrem = new PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.35;
    pmrem.dispose();

    this.scene.add(new HemisphereLight(0xffffff, 0x556070, 0.9));
    const key = new DirectionalLight(0xffffff, 2.4);
    key.position.set(-1.5, 2, 3);
    const fill = new DirectionalLight(0xdbe6ff, 0.9);
    fill.position.set(2.5, 0.5, 2);
    const rim = new DirectionalLight(0xffffff, 1.8);
    rim.position.set(0.5, 2, -3);
    this.scene.add(key, fill, rim);

    this.scene.add(this.pose);
    this.pose.add(this.calib);
    this.calib.add(this.fit);
    this.resize();
  }

  setQuality(q: number): void {
    if (q === this.quality) return;
    this.quality = q;
    this.resize();
  }

  resize(): void {
    const w = Math.max(1, window.innerWidth);
    const h = Math.max(1, window.innerHeight);
    this.renderer.setPixelRatio(pixelRatio(w, h, this.quality));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.updateCamera();
  }

  private updateCamera(): void {
    // Ensure the framed region also fits horizontally on narrow windows.
    const h = this.visibleH * Math.max(1, 1 / this.camera.aspect);
    const d = h / 2 / Math.tan((FOV * Math.PI) / 360);
    this.camera.position.set(0, this.targetY, d);
    this.camera.lookAt(0, this.targetY, 0);
    this.camera.updateProjectionMatrix();
  }

  /** Attach a model (disposing any previous one). Call refit() afterwards. */
  setModel(model: LoadedModel): void {
    this.clearModel();
    this.model = model;
    this.fit.add(model.root);
  }

  /**
   * Normalise size and framing. Avatars are scaled to unit height, standing on y=0 and framed
   * head-and-shoulders; everything else is scaled to a unit box centred on the origin.
   */
  refit(avatar: boolean, headY: () => number | null): void {
    if (!this.model) return;
    this.pose.position.set(0, 0, 0);
    this.pose.quaternion.identity();
    this.fit.position.set(0, 0, 0);
    this.fit.scale.setScalar(1);
    this.fit.updateMatrixWorld(true);

    this.box.setFromObject(this.model.root, true);
    const size = this.box.getSize(this.v);
    const c = this.box.getCenter(new Vector3());
    if (avatar) {
      const s = 1 / Math.max(size.y, 1e-4);
      this.fit.scale.setScalar(s);
      this.fit.position.set(-c.x * s, -this.box.min.y * s, -c.z * s);
    } else {
      const s = 1 / Math.max(size.x, size.y, size.z, 1e-4);
      this.fit.scale.setScalar(s);
      this.fit.position.set(-c.x * s, -c.y * s, -c.z * s);
    }
    this.fit.updateMatrixWorld(true);

    if (avatar) {
      const hy = headY() ?? 0.9;
      this.visibleH = 0.62;
      this.targetY = hy - 0.1;
    } else {
      this.visibleH = 1.5;
      this.targetY = 0;
    }
    this.updateCamera();
  }

  applyCalibration(m: ModelSettings): void {
    this.calib.position.set(m.px, m.py, m.pz);
    this.calib.scale.set((m.flipX ? -1 : 1) * m.scale, (m.flipY ? -1 : 1) * m.scale, (m.flipZ ? -1 : 1) * m.scale);
    const d = Math.PI / 180;
    this.eul.set(m.rx * d, m.ry * d, m.rz * d);
    this.calib.quaternion.setFromEuler(this.eul);
  }

  clearModel(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.pose.position.set(0, 0, 0);
    this.pose.quaternion.identity();
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  /** Render once and copy a centred square of the canvas into a PNG thumbnail. */
  async thumbnail(size = 256): Promise<Blob | undefined> {
    this.render();
    const src = this.canvas;
    const side = Math.min(src.width, src.height);
    if (side === 0) return undefined;
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const ctx = c.getContext('2d');
    if (!ctx) return undefined;
    ctx.drawImage(src, (src.width - side) / 2, (src.height - side) / 2, side, side, 0, 0, size, size);
    return new Promise((resolve) => c.toBlob((b) => resolve(b ?? undefined), 'image/png'));
  }
}
