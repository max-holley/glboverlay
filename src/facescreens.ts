import { CanvasTexture, FrontSide, MeshBasicMaterial, SRGBColorSpace, Vector3, type Material, type Mesh, type Object3D } from 'three';
import { normaliseName } from './arkit';
import type { FaceFrame } from './landmarker';

type Kind = 'eye' | 'mouth';

/** Mesh names that become live webcam "screens" showing the user's own eyes and mouth. */
const SCREENS: Record<string, Kind> = { faceeyel: 'eye', faceeyer: 'eye', facemouth: 'mouth' };

export function isFaceScreen(o: Object3D): boolean {
  return !!(o as Mesh).isMesh && normaliseName(o.name) in SCREENS;
}

interface Screen {
  mesh: Mesh;
  original: Material | Material[];
  material: MeshBasicMaterial;
  ctx: CanvasRenderingContext2D;
  mask: HTMLCanvasElement;
  tex: CanvasTexture;
  w: number;
  h: number;
  aspect: number;
}

const TEX_W = 256;
/** Per-tracking-frame smoothing of the crop centres (higher = snappier). */
const POINT_ALPHA = 0.55;

/**
 * Crops the user's eyes and mouth out of the webcam frame and draws them onto `FaceEyeL`,
 * `FaceEyeR` and `FaceMouth` meshes, so they turn, tilt and hide with the model.
 */
export class FaceScreens {
  /** `left` / `right` are the eye screens on the viewer's left / right. */
  private left: Screen | null = null;
  private right: Screen | null = null;
  private mouth: Screen | null = null;
  private tracked = false;
  private pts = new Map<string, { x: number; y: number }>();

  constructor(root: Object3D) {
    root.updateWorldMatrix(true, true);
    const eyes: { s: Screen; x: number }[] = [];
    const v = new Vector3();
    root.traverse((o) => {
      if (!isFaceScreen(o)) return;
      const mesh = o as Mesh;
      const kind = SCREENS[normaliseName(o.name)];
      const s = makeScreen(mesh, kind);
      if (kind === 'mouth') this.mouth ??= s;
      else {
        // Use the geometry centre: exporters often leave every mesh origin at the model origin.
        if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
        mesh.geometry.boundingBox!.getCenter(v).applyMatrix4(mesh.matrixWorld);
        eyes.push({ s, x: root.worldToLocal(v).x });
      }
    });
    // The model faces +Z, so the viewer's left is the model's -X.
    eyes.sort((a, b) => a.x - b.x);
    this.left = eyes[0]?.s ?? null;
    this.right = eyes[1]?.s ?? null;
  }

  get count(): number {
    return [this.left, this.right, this.mouth].filter(Boolean).length;
  }

  private smooth(key: string, x: number, y: number): { x: number; y: number } {
    const p = this.pts.get(key);
    if (!p) {
      const n = { x, y };
      this.pts.set(key, n);
      return n;
    }
    p.x += (x - p.x) * POINT_ALPHA;
    p.y += (y - p.y) * POINT_ALPHA;
    return p;
  }

  /** Redraw the screens from a new detection. */
  update(frame: FaceFrame | null, video: HTMLVideoElement, mirror: boolean): void {
    this.tracked = !!frame;
    if (!frame || this.count === 0) return;
    const L = frame.landmarks;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const px = (i: number) => L[i * 3] * vw;
    const py = (i: number) => L[i * 3 + 1] * vh;

    // Eye centres from their corners, in video pixels.
    const sL = this.smooth('l', (px(33) + px(133)) / 2, (py(33) + py(133)) / 2);
    const sR = this.smooth('r', (px(362) + px(263)) / 2, (py(362) + py(263)) / 2);
    const angle = Math.atan2(sR.y - sL.y, sR.x - sL.x);
    const iod = Math.hypot(sR.x - sL.x, sR.y - sL.y);
    // The screen on the viewer's left shows whichever eye appears on the left of the displayed image.
    const a = mirror ? sR : sL;
    const b = mirror ? sL : sR;
    const ew = iod * 0.66;
    if (this.left) drawScreen(this.left, video, a.x, a.y, ew, ew / this.left.aspect, angle, mirror);
    if (this.right) drawScreen(this.right, video, b.x, b.y, ew, ew / this.right.aspect, angle, mirror);
    if (this.mouth) {
      const m = this.smooth('m', (px(61) + px(291)) / 2, (py(13) + py(14)) / 2);
      const mw = Math.hypot(px(291) - px(61), py(291) - py(61));
      // ~1.75:1 crop, stretched to fill the screen's own shape.
      drawScreen(this.mouth, video, m.x, m.y, mw * 1.4, mw * 0.8, angle, mirror);
    }
  }

  /** Fade the screens in while a face is tracked and out when it is lost. Call every rendered frame. */
  tick(dt: number): void {
    const k = 1 - Math.pow(0.65, dt * 30);
    const target = this.tracked ? 1 : 0;
    for (const s of [this.left, this.right, this.mouth]) {
      if (s) s.material.opacity += (target - s.material.opacity) * k;
    }
  }

  dispose(): void {
    for (const s of [this.left, this.right, this.mouth]) {
      if (!s) continue;
      s.mesh.material = s.original;
      s.material.dispose();
      s.tex.dispose();
    }
    this.left = this.right = this.mouth = null;
  }
}

function makeScreen(mesh: Mesh, kind: Kind): Screen {
  // Aspect ratio of the screen from the positions at its UV corners.
  const g = mesh.geometry;
  const uv = g.attributes.uv;
  const pos = g.attributes.position;
  let aspect = kind === 'mouth' ? 4.5 : 1;
  if (uv) {
    let i00 = -1;
    let i10 = -1;
    let i01 = -1;
    for (let i = 0; i < uv.count; i++) {
      const u = uv.getX(i);
      const v = uv.getY(i);
      if (u < 0.01 && v < 0.01) i00 = i;
      if (u > 0.99 && v < 0.01) i10 = i;
      if (u < 0.01 && v > 0.99) i01 = i;
    }
    if (i00 >= 0 && i10 >= 0 && i01 >= 0) {
      const p = (i: number) => new Vector3().fromBufferAttribute(pos, i);
      const w = p(i00).distanceTo(p(i10));
      const h = p(i00).distanceTo(p(i01));
      if (w > 0 && h > 0) aspect = w / h;
    }
  }
  const w = TEX_W;
  const h = Math.max(8, Math.round(w / aspect));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;

  // Soft-edged mask matching the screen: rounded rectangle for eyes, rounded slot for the mouth.
  const mask = document.createElement('canvas');
  mask.width = w;
  mask.height = h;
  const mc = mask.getContext('2d')!;
  const fe = 4;
  const r = kind === 'mouth' ? h * 0.42 : w * 0.11;
  mc.filter = `blur(${fe * 0.6}px)`;
  mc.fillStyle = '#000';
  mc.beginPath();
  mc.roundRect(fe, fe, w - 2 * fe, h - 2 * fe, Math.max(0, Math.min(r - fe, (h - 2 * fe) / 2)));
  mc.fill();

  const tex = new CanvasTexture(canvas);
  tex.colorSpace = SRGBColorSpace;
  const material = new MeshBasicMaterial({
    map: tex,
    transparent: true,
    toneMapped: false,
    opacity: 0,
    side: FrontSide,
    polygonOffset: true,
    polygonOffsetFactor: -2,
  });
  const original = mesh.material;
  mesh.material = material;
  mesh.renderOrder = 2;
  return { mesh, original, material, ctx, mask, tex, w, h, aspect };
}

/** Draw a crop of the video centred on (cx, cy), de-rotated by `angle` so the feature stays level. */
function drawScreen(
  s: Screen,
  video: HTMLVideoElement,
  cx: number,
  cy: number,
  cropW: number,
  cropH: number,
  angle: number,
  mirror: boolean,
): void {
  const { ctx, w, h } = s;
  ctx.globalCompositeOperation = 'source-over';
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.scale(w / cropW, h / cropH);
  if (mirror) ctx.scale(-1, 1);
  ctx.rotate(-angle);
  ctx.drawImage(video, -cx, -cy);
  ctx.restore();
  ctx.globalCompositeOperation = 'destination-in';
  ctx.drawImage(s.mask, 0, 0);
  ctx.globalCompositeOperation = 'source-over';
  s.tex.needsUpdate = true;
}
