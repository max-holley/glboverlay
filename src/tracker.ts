import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';
import { Matrix4, Quaternion, Vector3 } from 'three';
import { ARKIT_COUNT, ARKIT_INDEX } from './arkit';
import { OneEuro } from './filters';
import type { CameraSettings } from './settings';

export type TrackerErrorKind = 'denied' | 'no-camera' | 'in-use' | 'unsupported' | 'model' | 'other';

export class TrackerError extends Error {
  constructor(
    public kind: TrackerErrorKind,
    message: string,
  ) {
    super(message);
  }
}

/** Latest filtered tracking output, written by the tracker and read by the render loop. */
export interface TrackState {
  valid: boolean;
  /** performance.now()/1000 of the last frame that contained a face */
  time: number;
  pos: Vector3;
  quat: Quaternion;
  blends: Float32Array;
}

export function createTrackState(): TrackState {
  return {
    valid: false,
    time: -1e9,
    pos: new Vector3(),
    quat: new Quaternion(),
    blends: new Float32Array(ARKIT_COUNT),
  };
}

const BLINK_IDX = new Set([ARKIT_INDEX.get('eyeBlinkLeft')!, ARKIT_INDEX.get('eyeBlinkRight')!]);

/** Converts raw MediaPipe results into smoothed pose + blendshape values. */
export class PoseProcessor {
  readonly state = createTrackState();
  private fp = [new OneEuro(), new OneEuro(), new OneEuro()];
  private fq = [new OneEuro(), new OneEuro(), new OneEuro(), new OneEuro()];
  private fb: OneEuro[] = Array.from({ length: ARKIT_COUNT }, () => new OneEuro());
  private m = new Matrix4();
  private s = new Vector3();
  private rawP = new Vector3();
  private rawQ = new Quaternion();
  private lastT = -1e9;
  private catIndex = new Map<string, number>();

  constructor() {
    this.setSmoothing(0.5);
  }

  setSmoothing(s: number): void {
    const t = 1 - Math.min(1, Math.max(0, s));
    const pose = 0.15 * Math.pow(40, t); // 0.15 .. 6 Hz
    const blend = 0.6 * Math.pow(25, t); // 0.6 .. 15 Hz
    for (const f of this.fp) Object.assign(f, { minCutoff: pose * 2, beta: 0.4 });
    for (const f of this.fq) Object.assign(f, { minCutoff: pose, beta: 0.6 });
    this.fb.forEach((f, i) => {
      const blink = BLINK_IDX.has(i);
      Object.assign(f, { minCutoff: blend * (blink ? 4 : 1), beta: blink ? 4 : 1.2 });
    });
  }

  process(result: FaceLandmarkerResult, t: number): boolean {
    const mat = result.facialTransformationMatrixes?.[0];
    const cats = result.faceBlendshapes?.[0]?.categories;
    if (!mat || !cats) return false;

    if (t - this.lastT > 0.5) this.resetFilters();
    this.lastT = t;

    const d = mat.data;
    this.m.fromArray(d);
    // Column-major has a zero bottom row in the rotation part; row-major puts the translation there.
    if (Math.abs(d[12]) + Math.abs(d[13]) + Math.abs(d[14]) < 1e-6 && Math.abs(d[3]) + Math.abs(d[7]) + Math.abs(d[11]) > 1e-6) {
      this.m.transpose();
    }
    const st = this.state;
    this.m.decompose(this.rawP, this.rawQ, this.s);
    if (st.valid && this.rawQ.dot(st.quat) < 0) this.rawQ.set(-this.rawQ.x, -this.rawQ.y, -this.rawQ.z, -this.rawQ.w);

    st.pos.set(this.fp[0].filter(this.rawP.x, t), this.fp[1].filter(this.rawP.y, t), this.fp[2].filter(this.rawP.z, t));
    st.quat
      .set(
        this.fq[0].filter(this.rawQ.x, t),
        this.fq[1].filter(this.rawQ.y, t),
        this.fq[2].filter(this.rawQ.z, t),
        this.fq[3].filter(this.rawQ.w, t),
      )
      .normalize();

    if (this.catIndex.size === 0) {
      for (const c of cats) {
        const i = ARKIT_INDEX.get(c.categoryName);
        if (i !== undefined) this.catIndex.set(c.categoryName, i);
      }
    }
    for (let i = 0; i < cats.length; i++) {
      const idx = this.catIndex.get(cats[i].categoryName);
      if (idx !== undefined) st.blends[idx] = this.fb[idx].filter(cats[i].score, t);
    }
    st.valid = true;
    st.time = t;
    return true;
  }

  private resetFilters(): void {
    for (const f of this.fp) f.reset();
    for (const f of this.fq) f.reset();
    for (const f of this.fb) f.reset();
  }
}

const BASE = new URL(import.meta.env.BASE_URL, location.href).href;

export interface TrackerStatus {
  running: boolean;
  delegate: string;
  fps: number;
}

/** Owns the camera stream, the hidden <video> and the Face Landmarker. */
export class Tracker {
  readonly processor = new PoseProcessor();
  onStatus: ((s: TrackerStatus) => void) | null = null;
  onError: ((e: TrackerError) => void) | null = null;
  onDevices: (() => void) | null = null;

  private landmarker: FaceLandmarker | null = null;
  private delegate = '';
  private stream: MediaStream | null = null;
  private running = false;
  private gen = 0;
  private lastVideoTime = -1;
  private frames = 0;
  private fpsT = 0;
  private fps = 0;

  constructor(private video: HTMLVideoElement) {
    navigator.mediaDevices?.addEventListener('devicechange', () => this.onDevices?.());
  }

  private async ensureLandmarker(): Promise<void> {
    if (this.landmarker) return;
    const fileset = await FilesetResolver.forVisionTasks(BASE + 'mediapipe/wasm');
    const make = (delegate: 'GPU' | 'CPU') =>
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: BASE + 'mediapipe/face_landmarker.task', delegate },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });
    try {
      this.landmarker = await make('GPU');
      this.delegate = 'GPU';
    } catch (e) {
      console.warn('GPU delegate failed, falling back to CPU', e);
      try {
        this.landmarker = await make('CPU');
        this.delegate = 'CPU';
      } catch (e2) {
        throw new TrackerError('model', `Could not start face tracking: ${(e2 as Error).message}`);
      }
    }
  }

  async start(cam: CameraSettings): Promise<void> {
    const gen = ++this.gen;
    this.stopStream();
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new TrackerError('unsupported', 'Camera access is not available (needs a secure context: https or localhost).');
    }
    const video: MediaTrackConstraints = {
      width: { ideal: cam.width },
      height: { ideal: cam.height },
      frameRate: { ideal: cam.fps },
    };
    if (cam.deviceId) video.deviceId = { exact: cam.deviceId };

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (e) {
      const err = e as DOMException;
      if (cam.deviceId && (err.name === 'OverconstrainedError' || err.name === 'NotFoundError')) {
        // Saved device vanished: fall back to the default camera.
        return this.start({ ...cam, deviceId: '' });
      }
      throw this.mapError(err);
    }
    if (gen !== this.gen) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    this.stream = stream;
    stream.getVideoTracks()[0].addEventListener('ended', () => {
      if (gen === this.gen) {
        this.running = false;
        this.emit();
        this.onError?.(new TrackerError('no-camera', 'Camera disconnected. Plug it back in to resume.'));
      }
    });
    this.video.srcObject = stream;
    await this.video.play().catch(() => undefined);
    this.onDevices?.();
    await this.ensureLandmarker();
    if (gen !== this.gen) return;
    this.processor.state.valid = false;
    this.running = true;
    this.lastVideoTime = -1;
    this.fpsT = performance.now();
    this.frames = 0;
    this.emit();
    this.loop(gen);
  }

  stop(): void {
    this.gen++;
    this.stopStream();
    this.emit();
  }

  get isRunning(): boolean {
    return this.running;
  }

  get activeDeviceId(): string {
    return this.stream?.getVideoTracks()[0]?.getSettings().deviceId ?? '';
  }

  private stopStream(): void {
    this.running = false;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  private mapError(err: DOMException): TrackerError {
    switch (err.name) {
      case 'NotAllowedError':
      case 'SecurityError':
        return new TrackerError(
          'denied',
          'Camera permission was denied. Allow camera access for this site in Chrome (lock icon in the address bar), then retry.',
        );
      case 'NotFoundError':
      case 'OverconstrainedError':
        return new TrackerError('no-camera', 'No camera was found. Connect a webcam and retry.');
      case 'NotReadableError':
      case 'AbortError':
        return new TrackerError('in-use', 'The camera could not be opened. Another app may be using it.');
      default:
        return new TrackerError('other', `Camera error: ${err.message || err.name}`);
    }
  }

  private loop(gen: number): void {
    const v = this.video;
    const step = () => {
      if (gen !== this.gen || !this.running) return;
      if (v.currentTime !== this.lastVideoTime && v.readyState >= 2) {
        this.lastVideoTime = v.currentTime;
        const t = performance.now();
        try {
          this.processor.process(this.landmarker!.detectForVideo(v, t), t / 1000);
        } catch (e) {
          console.warn('detect failed', e);
        }
        this.frames++;
        if (t - this.fpsT >= 1000) {
          this.fps = (this.frames * 1000) / (t - this.fpsT);
          this.frames = 0;
          this.fpsT = t;
          this.emit();
        }
      }
      schedule();
    };
    const schedule = () => {
      if ('requestVideoFrameCallback' in v) v.requestVideoFrameCallback(step);
      else requestAnimationFrame(step);
    };
    schedule();
  }

  private emit(): void {
    this.onStatus?.({ running: this.running, delegate: this.delegate, fps: this.fps });
  }

  async listDevices(): Promise<MediaDeviceInfo[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return [];
    return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  }
}
