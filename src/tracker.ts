import type { FaceLandmarker } from '@mediapipe/tasks-vision';
import { Matrix4, Quaternion, Vector3 } from 'three';
import type { WorkerRequest, WorkerResponse } from './detect.worker';
import type { FaceFrame } from './landmarker';
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

  process(f: FaceFrame, t: number): boolean {
    if (t - this.lastT > 0.5) this.resetFilters();
    this.lastT = t;

    const d = f.matrix;
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

    for (let i = 0; i < ARKIT_COUNT; i++) st.blends[i] = this.fb[i].filter(f.blends[i], t);
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

  /** Inference runs here when the worker is available; `landmarker` is the main-thread fallback. */
  private worker: Worker | null = null;
  private landmarker: FaceLandmarker | null = null;
  private toFrame: ((r: ReturnType<FaceLandmarker['detectForVideo']>) => FaceFrame | null) | null = null;
  private busy = false;
  private minInterval = 1000 / 15;
  private lastDetect = -1e9;
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
    if (this.worker || this.landmarker) return;
    const wasmBase = BASE + 'mediapipe/wasm';
    const modelUrl = BASE + 'mediapipe/face_landmarker.task';
    try {
      const { worker, delegate } = await this.startWorker(wasmBase, modelUrl);
      this.worker = worker;
      this.delegate = `${delegate}, worker`;
      return;
    } catch (e) {
      console.warn('Tracking worker unavailable, running on the main thread', e);
    }
    try {
      const lm = await import('./landmarker');
      const r = await lm.createLandmarker(wasmBase, modelUrl, false);
      this.landmarker = r.landmarker;
      this.toFrame = lm.toFrame;
      this.delegate = r.delegate;
    } catch (e) {
      throw new TrackerError('model', `Could not start face tracking: ${(e as Error).message}`);
    }
  }

  private startWorker(wasmBase: string, modelUrl: string): Promise<{ worker: Worker; delegate: string }> {
    return new Promise((resolve, reject) => {
      let worker: Worker;
      try {
        worker = new Worker(new URL('./detect.worker.ts', import.meta.url), { type: 'module' });
      } catch (e) {
        reject(e);
        return;
      }
      const fail = (err: unknown) => {
        worker.terminate();
        reject(err);
      };
      const timer = window.setTimeout(() => fail(new Error('worker init timed out')), 30000);
      worker.onerror = (e) => {
        window.clearTimeout(timer);
        fail(new Error(e.message || 'worker failed to load'));
      };
      worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
        const msg = e.data;
        if (msg.type === 'error') {
          window.clearTimeout(timer);
          fail(new Error(msg.message));
        } else if (msg.type === 'ready') {
          window.clearTimeout(timer);
          worker.onmessage = (ev: MessageEvent<WorkerResponse>) => this.onWorkerMessage(ev.data);
          worker.onerror = (ev) => {
            console.warn('tracking worker error', ev.message);
            this.busy = false;
          };
          resolve({ worker, delegate: msg.delegate });
        }
      };
      worker.postMessage({ type: 'init', wasmBase, modelUrl } satisfies WorkerRequest);
    });
  }

  private onWorkerMessage(msg: WorkerResponse): void {
    if (msg.type !== 'result') return;
    this.busy = false;
    if (!this.running) return;
    this.handleFrame(msg.frame, msg.t);
  }

  private handleFrame(frame: FaceFrame | null, t: number): void {
    if (frame) this.processor.process(frame, t / 1000);
    this.frames++;
    const now = performance.now();
    if (now - this.fpsT >= 1000) {
      this.fps = (this.frames * 1000) / (now - this.fpsT);
      this.frames = 0;
      this.fpsT = now;
      this.emit();
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

  /** Limit face detections per second; camera frames in between are skipped entirely. */
  setRate(fps: number): void {
    this.minInterval = 1000 / Math.max(1, fps);
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
      // Drop camera frames while the worker is still busy, so latency never builds up.
      const t = performance.now();
      // 3 ms slack so a 30 fps camera reliably yields 15 detections, not 10.
      if (!this.busy && t - this.lastDetect >= this.minInterval - 3 && v.currentTime !== this.lastVideoTime && v.readyState >= 2) {
        this.lastVideoTime = v.currentTime;
        this.lastDetect = t;
        if (this.worker) {
          const worker = this.worker;
          this.busy = true;
          createImageBitmap(v).then(
            (image) => {
              if (gen !== this.gen) {
                image.close();
                this.busy = false;
                return;
              }
              worker.postMessage({ type: 'frame', image, t } satisfies WorkerRequest, [image]);
            },
            (e) => {
              console.warn('frame capture failed', e);
              this.busy = false;
            },
          );
        } else {
          let frame: FaceFrame | null = null;
          try {
            frame = this.toFrame!(this.landmarker!.detectForVideo(v, t));
          } catch (e) {
            console.warn('detect failed', e);
          }
          this.handleFrame(frame, t);
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
