import { Euler, Object3D, Quaternion, Vector3, type Bone, type Mesh } from 'three';
import type { VRM } from '@pixiv/three-vrm';
import { ARKIT_COUNT, ARKIT_INDEX, ARKIT_MIRROR, ARKIT_NORMALISED, normaliseName } from './arkit';
import type { LoadedModel } from './loader';
import type { GlobalSettings, ModelSettings } from './settings';
import type { TrackState } from './tracker';

const HOLD_SECONDS = 0.35;
const FADE_OUT_SECONDS = 0.7;
const FADE_IN_SECONDS = 0.12;
const NECK_SHARE = 0.35;
/** scene units per centimetre of head translation */
const MOVE_SCALE = 0.02;
const MAX_ANGLE = Math.PI * 0.85;

interface BoneRef {
  bone: Object3D;
  restLocal: Quaternion;
  restWorld: Quaternion;
  restWorldInv: Quaternion;
}

interface MorphBinding {
  mesh: Mesh;
  index: number;
  arkit: number;
}

type Role = 'head' | 'neck' | 'jaw' | 'leftEye' | 'rightEye';

const IDX = (n: string) => ARKIT_INDEX.get(n)!;
const I = {
  blinkL: IDX('eyeBlinkLeft'),
  blinkR: IDX('eyeBlinkRight'),
  jawOpen: IDX('jawOpen'),
  funnel: IDX('mouthFunnel'),
  pucker: IDX('mouthPucker'),
  stretchL: IDX('mouthStretchLeft'),
  stretchR: IDX('mouthStretchRight'),
  smileL: IDX('mouthSmileLeft'),
  smileR: IDX('mouthSmileRight'),
  frownL: IDX('mouthFrownLeft'),
  frownR: IDX('mouthFrownRight'),
  browDownL: IDX('browDownLeft'),
  browDownR: IDX('browDownRight'),
  browInner: IDX('browInnerUp'),
  browOuterL: IDX('browOuterUpLeft'),
  browOuterR: IDX('browOuterUpRight'),
  wideL: IDX('eyeWideLeft'),
  wideR: IDX('eyeWideRight'),
  sneerL: IDX('noseSneerLeft'),
  sneerR: IDX('noseSneerRight'),
  inL: IDX('eyeLookInLeft'),
  inR: IDX('eyeLookInRight'),
  outL: IDX('eyeLookOutLeft'),
  outR: IDX('eyeLookOutRight'),
  upL: IDX('eyeLookUpLeft'),
  upR: IDX('eyeLookUpRight'),
  downL: IDX('eyeLookDownLeft'),
  downR: IDX('eyeLookDownRight'),
};

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

function boneRole(name: string): Role | null {
  const n = normaliseName(name.replace(/^mixamorig:?/i, ''));
  switch (n) {
    case 'head':
      return 'head';
    case 'neck':
    case 'neck1':
      return 'neck';
    case 'jaw':
      return 'jaw';
    case 'lefteye':
    case 'eyeleft':
    case 'eyel':
    case 'leye':
      return 'leftEye';
    case 'righteye':
    case 'eyeright':
    case 'eyer':
    case 'reye':
      return 'rightEye';
    default:
      return null;
  }
}

export interface RigInfo {
  detectedMode: 'avatar' | 'rigid';
  morphTargets: number;
  vrm: boolean;
  perfectSync: boolean;
  jawBone: boolean;
  eyeBones: boolean;
}

/** Maps tracking data onto one loaded model, either as an avatar (bones + expressions) or a rigid prop. */
export class ModelRig {
  readonly info: RigInfo;
  mode: 'avatar' | 'rigid';

  private vrm: VRM | null;
  private bones: Partial<Record<Role, BoneRef>> = {};
  private morphs: MorphBinding[] = [];
  private exprNames = new Map<string, string>();
  private perfectSync = false;
  private exprArkit: { idx: number; name: string }[] = [];
  private gaze: Object3D | null = null;
  private hasMouthExpr = false;

  private weight = 0;
  private neutralQ = new Quaternion();
  private neutralQInv = new Quaternion();
  private neutralP = new Vector3();
  private hasNeutral = false;

  // scratch
  private d = new Quaternion();
  private dHead = new Quaternion();
  private dNeck = new Quaternion();
  private q = new Quaternion();
  private dp = new Vector3();
  private v3 = new Vector3();
  private b = new Float32Array(ARKIT_COUNT);

  onNeutralChange: ((n: { q: number[]; p: number[] }) => void) | null = null;

  constructor(
    private model: LoadedModel,
    host: Object3D,
    overrideMode: ModelSettings['mode'],
    neutral?: ModelSettings['neutral'],
  ) {
    this.vrm = model.vrm;
    model.root.updateWorldMatrix(true, true);
    this.collectBones();
    this.collectMorphs();
    this.collectExpressions();

    const detected = this.bones.head && (this.vrm || this.looksHumanoid()) ? 'avatar' : 'rigid';
    this.mode = overrideMode === 'auto' ? detected : overrideMode;
    this.info = {
      detectedMode: detected,
      morphTargets: this.morphs.length,
      vrm: !!this.vrm,
      perfectSync: this.perfectSync,
      jawBone: !!this.bones.jaw,
      eyeBones: !!(this.bones.leftEye && this.bones.rightEye),
    };

    if (this.vrm?.lookAt && !this.perfectSync) {
      this.gaze = new Object3D();
      host.add(this.gaze);
      this.vrm.lookAt.target = this.gaze;
    }
    if (neutral) {
      this.neutralQ.fromArray(neutral.q);
      this.neutralP.fromArray(neutral.p);
      this.neutralQInv.copy(this.neutralQ).invert();
      this.hasNeutral = true;
    }
  }

  private looksHumanoid(): boolean {
    let skeletal = 0;
    this.model.root.traverse((o) => {
      if ((o as Bone).isBone) {
        const n = normaliseName(o.name.replace(/^mixamorig:?/i, ''));
        if (n === 'hips' || n === 'pelvis' || n.startsWith('spine')) skeletal++;
      }
    });
    return skeletal > 0;
  }

  private ref(bone: Object3D): BoneRef {
    const restWorld = bone.getWorldQuaternion(new Quaternion());
    return { bone, restLocal: bone.quaternion.clone(), restWorld, restWorldInv: restWorld.clone().invert() };
  }

  private collectBones(): void {
    const set = (role: Role, node: Object3D | null | undefined) => {
      if (node && !this.bones[role]) this.bones[role] = this.ref(node);
    };
    if (this.vrm) {
      const h = this.vrm.humanoid;
      set('head', h.getNormalizedBoneNode('head'));
      set('neck', h.getNormalizedBoneNode('neck'));
      set('jaw', h.getNormalizedBoneNode('jaw'));
      set('leftEye', h.getNormalizedBoneNode('leftEye'));
      set('rightEye', h.getNormalizedBoneNode('rightEye'));
      return;
    }
    this.model.root.traverse((o) => {
      if (!(o as Bone).isBone) return;
      const role = boneRole(o.name);
      if (role) set(role, o);
    });
  }

  private collectMorphs(): void {
    this.model.root.traverse((o) => {
      const m = o as Mesh;
      if (!m.isMesh || !m.morphTargetDictionary || !m.morphTargetInfluences) return;
      for (const [name, index] of Object.entries(m.morphTargetDictionary)) {
        const arkit = ARKIT_NORMALISED.get(normaliseName(name));
        if (arkit !== undefined) this.morphs.push({ mesh: m, index, arkit });
      }
    });
  }

  private collectExpressions(): void {
    const mgr = this.vrm?.expressionManager;
    if (!mgr) return;
    for (const name of Object.keys(mgr.expressionMap)) this.exprNames.set(name.toLowerCase(), name);
    for (const [norm, idx] of ARKIT_NORMALISED) {
      const name = this.exprNames.get(norm);
      if (name) this.exprArkit.push({ idx, name });
    }
    this.perfectSync = this.exprArkit.length >= 20;
    this.hasMouthExpr = ['aa', 'ih', 'ou', 'ee', 'oh'].some((n) => this.exprNames.has(n));
  }

  /** World-space Y of the head in the fitted scene, for camera framing. */
  headWorldY(): number | null {
    const h = this.bones.head?.bone;
    if (!h) return null;
    h.updateWorldMatrix(true, false);
    return h.getWorldPosition(this.v3).y;
  }

  recalibrate(state: TrackState): boolean {
    if (!state.valid) return false;
    this.neutralQ.copy(state.quat);
    this.neutralQInv.copy(state.quat).invert();
    this.neutralP.copy(state.pos);
    this.hasNeutral = true;
    this.emitNeutral();
    return true;
  }

  private emitNeutral(): void {
    this.onNeutralChange?.({ q: this.neutralQ.toArray(), p: this.neutralP.toArray() });
  }

  /** Apply a delta rotation given in world space to a bone, preserving its rest orientation. */
  private applyDelta(r: BoneRef | undefined, dWorld: Quaternion): void {
    if (!r) return;
    this.q.copy(r.restWorldInv).multiply(dWorld).multiply(r.restWorld);
    r.bone.quaternion.copy(r.restLocal).multiply(this.q);
  }

  update(
    dt: number,
    now: number,
    st: TrackState,
    g: GlobalSettings,
    pose: Object3D,
  ): void {
    // Presence weight: hold briefly when the face is lost, then ease to neutral.
    const present = st.valid && now - st.time < HOLD_SECONDS;
    if (present) this.weight = Math.min(1, this.weight + dt / FADE_IN_SECONDS);
    else if (st.valid) this.weight = Math.max(0, this.weight - dt / FADE_OUT_SECONDS);
    else this.weight = 0;
    const w = this.weight;

    if (st.valid && !this.hasNeutral) this.recalibrate(st);
    const mirror = g.mirror;

    // ---- pose ----
    this.d.copy(st.quat).multiply(this.neutralQInv);
    if (mirror) this.d.set(this.d.x, -this.d.y, -this.d.z, this.d.w);
    this.dp.copy(st.pos).sub(this.neutralP);
    if (mirror) this.dp.x = -this.dp.x;

    const sens = g.headSensitivity * w;
    if (this.mode === 'rigid') {
      scaleRotation(this.d, sens, this.dHead);
      pose.quaternion.copy(this.dHead);
      pose.position.copy(this.dp).multiplyScalar(MOVE_SCALE * g.headSensitivity * w);
      pose.position.z *= 0.5;
    } else {
      pose.quaternion.identity();
      pose.position.copy(this.dp).multiplyScalar(MOVE_SCALE * 0.4 * g.headSensitivity * w);
      pose.position.z *= 0.3;
      const hasNeck = !!this.bones.neck;
      scaleRotation(this.d, sens * (hasNeck ? 1 - NECK_SHARE : 1), this.dHead);
      this.applyDelta(this.bones.head, this.dHead);
      if (hasNeck) {
        scaleRotation(this.d, sens * NECK_SHARE, this.dNeck);
        this.applyDelta(this.bones.neck, this.dNeck);
      }
    }

    // ---- expression values (mirrored, weighted, intensity-scaled) ----
    const src = st.blends;
    const k = g.expression * w;
    for (let i = 0; i < ARKIT_COUNT; i++) this.b[i] = clamp01(src[mirror ? ARKIT_MIRROR[i] : i] * k);
    const b = this.b;

    for (const m of this.morphs) m.mesh.morphTargetInfluences![m.index] = b[m.arkit];

    const vrm = this.vrm;
    const mgr = vrm?.expressionManager;
    if (mgr) {
      for (const e of this.exprArkit) mgr.setValue(e.name, b[e.idx]);
      if (!this.perfectSync) this.driveVrmPresets(b);
    }

    if (this.mode === 'avatar') {
      // Jaw bone fallback: only when nothing else will open the mouth.
      const jaw = this.bones.jaw;
      if (jaw && !this.hasJawMorph() && !this.hasMouthExpr) {
        this.q.setFromAxisAngle(AXIS_X, b[I.jawOpen] * 0.38);
        this.applyDelta(jaw, this.q);
      }
      this.driveGaze(b);
    }

    if (vrm) vrm.update(dt);
  }

  private hasJawMorphCache: boolean | null = null;
  private hasJawMorph(): boolean {
    if (this.hasJawMorphCache === null) this.hasJawMorphCache = this.morphs.some((m) => m.arkit === I.jawOpen);
    return this.hasJawMorphCache;
  }

  private driveVrmPresets(b: Float32Array): void {
    const mgr = this.vrm!.expressionManager!;
    const has = (n: string) => this.exprNames.has(n.toLowerCase());
    const set = (n: string, v: number) => {
      const name = this.exprNames.get(n.toLowerCase());
      if (name) mgr.setValue(name, clamp01(v));
    };

    if (has('blinkLeft') && has('blinkRight')) {
      set('blinkLeft', b[I.blinkL]);
      set('blinkRight', b[I.blinkR]);
      set('blink', 0);
    } else {
      set('blink', (b[I.blinkL] + b[I.blinkR]) / 2);
    }

    const jaw = b[I.jawOpen];
    const funnel = b[I.funnel];
    const pucker = b[I.pucker];
    const stretch = (b[I.stretchL] + b[I.stretchR]) / 2;
    const smile = (b[I.smileL] + b[I.smileR]) / 2;
    let aa = jaw * (1 - 0.6 * pucker);
    let oh = funnel * (0.4 + 0.6 * jaw);
    let ou = pucker * (1 - 0.5 * jaw);
    let ee = stretch * 0.8 + smile * 0.3 * (1 - jaw);
    let ih = ee * 0.5 * (1 - jaw);
    const sum = aa + oh + ou + ee + ih;
    if (sum > 1) {
      aa /= sum;
      oh /= sum;
      ou /= sum;
      ee /= sum;
      ih /= sum;
    }
    set('aa', aa);
    set('oh', oh);
    set('ou', ou);
    set('ee', ee);
    set('ih', ih);

    const frown = (b[I.frownL] + b[I.frownR]) / 2;
    const browDown = (b[I.browDownL] + b[I.browDownR]) / 2;
    const sneer = (b[I.sneerL] + b[I.sneerR]) / 2;
    const wide = (b[I.wideL] + b[I.wideR]) / 2;
    set('happy', smile * 0.7);
    set('angry', browDown * 0.7 + sneer * 0.3);
    set('sad', b[I.browInner] * 0.5 + frown * 0.6);
    set('surprised', wide * 0.5 + b[I.browInner] * 0.3 + jaw * 0.2);
  }

  private driveGaze(b: Float32Array): void {
    // b[] is already side-swapped when mirroring. Positive = avatar looks to its own left (+X).
    const lookLeft = (b[I.outL] + b[I.inR]) / 2 - (b[I.inL] + b[I.outR]) / 2;
    const lookUp = (b[I.upL] + b[I.upR]) / 2 - (b[I.downL] + b[I.downR]) / 2;
    const gx = lookLeft;

    const head = this.bones.head?.bone;
    if (this.gaze && head) {
      head.updateWorldMatrix(true, false);
      head.getWorldPosition(this.v3);
      this.dir.set(gx * 0.9, lookUp * 0.6, 1).applyQuaternion(this.dHead);
      this.gaze.position.copy(this.v3).addScaledVector(this.dir, 1.5);
      this.gaze.updateMatrixWorld(true);
      return;
    }
    const l = this.bones.leftEye;
    const r = this.bones.rightEye;
    if (l || r) {
      this.q.setFromEuler(this.eul.set(-lookUp * 0.4, gx * 0.45, 0));
      this.applyDelta(l, this.q);
      this.applyDelta(r, this.q);
    }
  }

  private dir = new Vector3();
  private eul = new Euler();

  dispose(): void {
    this.gaze?.removeFromParent();
    if (this.vrm?.lookAt) this.vrm.lookAt.target = null;
  }
}

const AXIS_X = new Vector3(1, 0, 0);

/** Scale the rotation angle of `q` by `s`, writing to `out` (no allocation). */
function scaleRotation(q: Quaternion, s: number, out: Quaternion): void {
  let { x, y, z, w } = q;
  if (w < 0) {
    x = -x;
    y = -y;
    z = -z;
    w = -w;
  }
  w = Math.min(1, w);
  const sin = Math.sqrt(1 - w * w);
  if (sin < 1e-6 || s === 0) {
    out.identity();
    return;
  }
  const angle = Math.min(2 * Math.acos(w) * s, MAX_ANGLE);
  const half = angle / 2;
  const f = Math.sin(half) / sin;
  out.set(x * f, y * f, z * f, Math.cos(half));
}
