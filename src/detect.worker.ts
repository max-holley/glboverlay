/// <reference lib="webworker" />
// Runs the Face Landmarker off the main thread so inference never blocks rendering.
import type { FaceLandmarker } from '@mediapipe/tasks-vision';
import { createLandmarker, toFrame, type FaceFrame } from './landmarker';

export type WorkerRequest =
  | { type: 'init'; wasmBase: string; modelUrl: string }
  | { type: 'frame'; image: ImageBitmap; t: number };

export type WorkerResponse =
  | { type: 'ready'; delegate: string }
  | { type: 'error'; message: string }
  | { type: 'result'; t: number; frame: FaceFrame | null };

const scope = self as unknown as DedicatedWorkerGlobalScope;
let landmarker: FaceLandmarker | null = null;

const post = (msg: WorkerResponse, transfer: Transferable[] = []) => scope.postMessage(msg, transfer);

scope.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  if (msg.type === 'init') {
    try {
      const r = await createLandmarker(msg.wasmBase, msg.modelUrl, true);
      landmarker = r.landmarker;
      post({ type: 'ready', delegate: r.delegate });
    } catch (err) {
      post({ type: 'error', message: (err as Error)?.message ?? String(err) });
    }
    return;
  }
  let frame: FaceFrame | null = null;
  try {
    if (landmarker) frame = toFrame(landmarker.detectForVideo(msg.image, msg.t));
  } catch (err) {
    console.warn('detect failed', err);
  } finally {
    msg.image.close();
  }
  post({ type: 'result', t: msg.t, frame }, frame ? [frame.matrix.buffer, frame.blends.buffer] : []);
};
