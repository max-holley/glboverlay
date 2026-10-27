import { Box3, Euler, Matrix4, Object3D, Quaternion, Vector3, type Bone, type Mesh, type SkinnedMesh } from 'three';
import type { VRM } from '@pixiv/three-vrm';
import { ARKIT_COUNT, ARKIT_INDEX, ARKIT_MIRROR, ARKIT_NORMALISED, normaliseName } from './arkit';
import { isFaceScreen } from './facescreens';
import type { LoadedModel } from './loader';
import type { GlobalSettings, ModelSettings } from './settings';
import type { TrackState } from './tracker';

/** Keep the last pose this long when the face is lost, so brief dropouts don't show. */
const HOLD_SECONDS = 2.5;
/** Time constant of the per-render-frame easing toward the latest tracked values. */
const EASE_SECONDS = 0.045;
const FADE_OUT_SECONDS = 0.7;
const FADE_IN_SECONDS = 0.12;
const NECK_SHARE = 0.35;
/** scene units per centimetre of head translation */
const MOVE_SCALE = 0.02;
const MAX_ANGLE = Math.PI * 0.85;

interface BoneRef {
  bone: Object3D;
  restLocal: Quaternion;
  /** Rest orientation relative to the model's parent (the stage's fit group), so calibration changes don't stale it. */
  restRel: Quaternion;
  restScale: Vector3;
  /** Local axes most aligned with the model's up (Y) and right (X), for squash/stretch. */
  upAxis: Axis;
  sideAxis: Axis;
}

type Axis = 'x' | 'y' | 'z';

interface MorphBinding {
  mesh: Mesh;
  index: number;
  arkit: number;
}

type Role = 'head' | 'neck' | 'jaw' | 'leftEye' | 'rightEye' | 'eyes' | 'mouth';

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

/** Name tokens that may accompany a part name without changing its meaning (rig prefixes, suffixes, counters). */
const FILLER = /^(\d+|j|jnt|joint|bone|bn|def|adj|bip\d*|c|cc|base|face|rig|armature|mesh|geo|geom|obj|grp|ball|lower|root|l|r|left|right)$/;

function tokens(name: string): string[] {
  return name
    .replace(/^mixamorig\d*:?/i, '')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
}

/** Guess a facial/head role from a node name: `Head`, `neck_01`, `Jaw`, `Eye.L`, `LeftEye`, `EyeBall_R`, `Eyes`, `Mouth`... */
export function partRole(name: string): Role | null {
  const t = tokens(name);
  let side: 'l' | 'r' | null = null;
  let part: string | null = null;
  for (const w of t) {
    if (w === 'l' || w === 'left') side = side === 'r' ? null : 'l';
    else if (w === 'r' || w === 'right') side = side === 'l' ? null : 'r';
    if (/^(head|neck\d*|jaw|jawbone|eye|eyeball|eyes|mouth)$/.test(w)) {
      if (part) return null;
      part = w;
    } else if (!FILLER.test(w)) {
      return null;
    }
  }
  if (!part) return null;
  if (part === 'head') return side ? null : 'head';
  if (part.startsWith('neck')) return side ? null : 'neck';
  if (part.startsWith('jaw')) return side ? null : 'jaw';
  if (part === 'mouth') return side ? null : 'mouth';
  if (side === 'l') return 'leftEye';
  if (side === 'r') return 'rightEye';
  return 'eyes';
}

export interface RigInfo {
  detectedMode: 'avatar' | 'rigid';
  morphTargets: number;
  vrm: boolean;
  perfectSync: boolean;
  jawBone: boolean;
  eyeBones: boolean;
  /** Names of the nodes found for eyes / jaw / mouth, for the settings panel. */
  faceParts: string[];
}

/** Maps tracking data onto one loaded model, either as an avatar (bones + expressions) or a rigid prop. */
export class ModelRig {
  readonly info: RigInfo;
  mode: 'avatar' | 'rigid';

  private vrm: VRM | null;
  private bones: Partial<Record<Role, BoneRef>> = {};
  /** Children of a combined `Eyes` node, each driven as one eye. */
  private eyeGroup: BoneRef[] = [];
  private morphs: MorphBinding[] = [];
  private morphArkit = new Set<number>();
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
  private rest = new Quaternion();
  private restInv = new Quaternion();
  /** applyDelta's own scratch: callers pass `q` as the delta, so it must not be written there. */
  private local = new Quaternion();
  /** Model frame (calibration rotation, excluding the rigid head pose), refreshed every update. */
  private frame = new Quaternion();
  private dp = new Vector3();
  private v3 = new Vector3();
  private b = new Float32Array(ARKIT_COUNT);
  // Eased tracking values: updated every rendered frame, so motion stays smooth between detections.
  private eq = new Quaternion();
  private ep = new Vector3();
  private eb = new Float32Array(ARKIT_COUNT);
  private eased = false;

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
    const partNames = (['leftEye', 'rightEye', 'eyes', 'jaw', 'mouth'] as const)
      .map((r) => this.bones[r]?.bone.name)
      .concat(this.eyeGroup.map((e) => e.bone.name))
      .filter((n): n is string => !!n);
    this.info = {
      detectedMode: detected,
      morphTargets: this.morphs.length,
      vrm: !!this.vrm,
      perfectSync: this.perfectSync,
      jawBone: !!this.bones.jaw,
      eyeBones: !!(this.bones.leftEye && this.bones.rightEye),
      faceParts: partNames,
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
    const parentQ = this.model.root.parent?.getWorldQuaternion(new Quaternion()) ?? new Quaternion();
    const restRel = parentQ.invert().multiply(bone.getWorldQuaternion(new Quaternion()));
    const inv = restRel.clone().invert();
    return {
      bone,
      restLocal: bone.quaternion.clone(),
      restRel,
      restScale: bone.scale.clone(),
      upAxis: dominantAxis(new Vector3(0, 1, 0).applyQuaternion(inv)),
      sideAxis: dominantAxis(new Vector3(1, 0, 0).applyQuaternion(inv)),
    };
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
    // Skinned joints first, then plain nodes: GLBs without a skin (separate eye / mouth meshes
    // parented to a head) have no Bone objects at all, only named nodes.
    const joints: [Role, Object3D][] = [];
    const plain: [Role, Object3D][] = [];
    this.model.root.traverse((o) => {
      if ((o as SkinnedMesh).isSkinnedMesh || isFaceScreen(o)) return;
      const role = partRole(o.name);
      if (role) ((o as Bone).isBone ? joints : plain).push([role, o]);
    });
    for (const [role, o] of [...joints, ...plain]) {
      if (this.bones[role]) continue;
      // Eye and mouth meshes often keep their origin at the model origin; pivot them on their own centre.
      if (!(o as Bone).isBone && (role === 'leftEye' || role === 'rightEye' || role === 'eyes' || role === 'mouth')) {
        recentre(o);
      }
      set(role, o);
    }
    const eyes = this.bones.eyes;
    if (this.bones.leftEye || this.bones.rightEye) {
      delete this.bones.eyes;
    } else if (eyes && !(eyes.bone as Bone).isBone && eyes.bone.children.length >= 2) {
      // An `Eyes` group holding the eyeballs: drive each one so they turn in place.
      delete this.bones.eyes;
      for (const child of eyes.bone.children) {
        recentre(child);
        this.eyeGroup.push(this.ref(child));
      }
    }
  }

  private collectMorphs(): void {
    this.model.root.traverse((o) => {
      const m = o as Mesh;
      if (!m.isMesh || !m.morphTargetDictionary || !m.morphTargetInfluences) return;
      for (const [name, index] of Object.entries(m.morphTargetDictionary)) {
        const arkit = ARKIT_NORMALISED.get(normaliseName(name));
        if (arkit !== undefined) {
          this.morphs.push({ mesh: m, index, arkit });
          this.morphArkit.add(arkit);
        }
      }
    });
  }

  private collectExpressions(): void {
    const mgr = this.vrm?.expressionManager;
    if (!mgr) return;
    for (const name of Object.keys(mgr.expressionMap)) this.exprNames.set(name.toLowerCase(), name);
    for (const [norm, idx] of ARKIT_NORMALISED) {
      const name = this.exprNames.get(norm);
      if (name) {
        this.exprArkit.push({ idx, name });
        this.morphArkit.add(idx);
      }
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

  /** Apply a delta rotation, given in the model frame, to a bone while preserving its rest orientation. */
  private applyDelta(r: BoneRef | undefined, dWorld: Quaternion): void {
    if (!r) return;
    this.rest.copy(this.frame).multiply(r.restRel);
    this.restInv.copy(this.rest).invert();
    this.local.copy(this.restInv).multiply(dWorld).multiply(this.rest);
    r.bone.quaternion.copy(r.restLocal).multiply(this.local);
  }

  /** Scale a part along its up / side axes relative to its rest scale. */
  private applyScale(r: BoneRef, up: number, side: number): void {
    r.bone.scale.copy(r.restScale);
    r.bone.scale[r.upAxis] *= up;
    if (r.sideAxis !== r.upAxis) r.bone.scale[r.sideAxis] *= side;
  }

  /** Rotation of the model's frame below the pose group (calibration), so deltas follow the model, not the world. */
  private updateFrame(pose: Object3D): void {
    this.frame.identity();
    for (let o = this.model.root.parent; o && o !== pose; o = o.parent) this.frame.premultiply(o.quaternion);
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
    this.updateFrame(pose);

    // ---- ease toward the latest detection ----
    if (st.valid && !this.eased) {
      this.eq.copy(st.quat);
      this.ep.copy(st.pos);
      this.eb.set(st.blends);
      this.eased = true;
    } else if (st.valid) {
      const ek = 1 - Math.exp(-dt / EASE_SECONDS);
      this.eq.slerp(st.quat, ek);
      this.ep.lerp(st.pos, ek);
      for (let i = 0; i < ARKIT_COUNT; i++) this.eb[i] += (st.blends[i] - this.eb[i]) * ek;
    }

    // ---- pose ----
    this.d.copy(this.eq).multiply(this.neutralQInv);
    if (mirror) this.d.set(this.d.x, -this.d.y, -this.d.z, this.d.w);
    this.dp.copy(this.ep).sub(this.neutralP);
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
    const src = this.eb;
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

    // Eye / jaw / mouth parts work in both modes; in rigid mode they move relative to the posed model.
    this.driveMouth(b);
    this.driveEyes(b);

    if (vrm) vrm.update(dt);
  }

  /** Jaw bone and mouth part fallback: only when no morph or expression opens the mouth. */
  private driveMouth(b: Float32Array): void {
    if (this.morphArkit.has(I.jawOpen) || this.hasMouthExpr) return;
    const jawOpen = b[I.jawOpen];
    const jaw = this.bones.jaw;
    if (jaw) {
      this.q.setFromAxisAngle(AXIS_X, jawOpen * 0.38);
      this.applyDelta(jaw, this.q);
    }
    const mouth = this.bones.mouth;
    if (mouth) {
      const smile = (b[I.smileL] + b[I.smileR]) / 2;
      const stretch = (b[I.stretchL] + b[I.stretchR]) / 2;
      const narrow = Math.max(b[I.pucker], b[I.funnel]);
      // Without a jaw, open the mouth by stretching it vertically; with one, only shape its width.
      const up = jaw ? 1 : 1 + 2.2 * jawOpen + 0.6 * b[I.funnel];
      const side = 1 + 0.3 * Math.max(smile, stretch) - 0.4 * narrow;
      this.applyScale(mouth, up, side);
    }
  }

  private driveEyes(b: Float32Array): void {
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
    // A single combined eyes mesh only blinks: rotating it would swing both eyes around their midpoint.
    const both = this.bones.eyes;
    const group = this.eyeGroup;
    if (!l && !r && !both && group.length === 0) return;

    if (!this.morphArkit.has(I.inL) && !this.morphArkit.has(I.upL)) {
      this.q.setFromEuler(this.eul.set(-lookUp * 0.4, gx * 0.45, 0));
      this.applyDelta(l, this.q);
      this.applyDelta(r, this.q);
      for (const e of group) this.applyDelta(e, this.q);
    }
    // Blink by squashing the eyes vertically when nothing else closes them.
    if (!this.morphArkit.has(I.blinkL) && !this.exprNames.has('blink') && !this.exprNames.has('blinkleft')) {
      const open = (v: number) => 1 - 0.9 * v;
      if (l) this.applyScale(l, open(b[I.blinkL]), 1);
      if (r) this.applyScale(r, open(b[I.blinkR]), 1);
      const avg = open((b[I.blinkL] + b[I.blinkR]) / 2);
      if (both) this.applyScale(both, avg, 1);
      for (const e of group) this.applyScale(e, avg, 1);
    }
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

  private dir = new Vector3();
  private eul = new Euler();

  dispose(): void {
    this.gaze?.removeFromParent();
    if (this.vrm?.lookAt) this.vrm.lookAt.target = null;
    // Put driven parts back at rest, so a rebuilt rig (mode switch) captures the true rest pose.
    for (const r of [...Object.values(this.bones), ...this.eyeGroup]) {
      r.bone.quaternion.copy(r.restLocal);
      r.bone.scale.copy(r.restScale);
    }
  }
}

const AXIS_X = new Vector3(1, 0, 0);

function dominantAxis(v: Vector3): Axis {
  const x = Math.abs(v.x);
  const y = Math.abs(v.y);
  const z = Math.abs(v.z);
  return x >= y && x >= z ? 'x' : y >= z ? 'y' : 'z';
}

/**
 * Move a node's pivot to the centre of its meshes without moving anything visually. Exporters often
 * leave eye / mouth meshes with their origin at the model origin, which would make them orbit instead
 * of rotate or squash in place. Skinned geometry is left alone.
 */
function recentre(node: Object3D): void {
  node.updateWorldMatrix(true, true);
  const inv = new Matrix4().copy(node.matrixWorld).invert();
  const m = new Matrix4();
  const box = new Box3();
  const part = new Box3();
  let skinned = false;
  node.traverse((o) => {
    const mesh = o as Mesh;
    if (!mesh.isMesh) return;
    if ((mesh as SkinnedMesh).isSkinnedMesh) skinned = true;
    if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
    box.union(part.copy(mesh.geometry.boundingBox!).applyMatrix4(m.multiplyMatrices(inv, mesh.matrixWorld)));
  });
  if (skinned || box.isEmpty()) return;
  const c = box.getCenter(new Vector3());
  const size = box.getSize(new Vector3());
  if (c.length() < 0.2 * Math.max(size.x, size.y, size.z)) return;

  const mesh = node as Mesh;
  if (mesh.isMesh) {
    mesh.geometry = mesh.geometry.clone();
    mesh.geometry.translate(-c.x, -c.y, -c.z);
  }
  for (const child of node.children) child.position.sub(c);
  node.position.add(c.multiply(node.scale).applyQuaternion(node.quaternion));
  node.updateMatrixWorld(true);
}

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
