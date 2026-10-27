import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';
import { ARKIT_COUNT, ARKIT_INDEX } from './arkit';

/** One detected face: raw 4x4 transform, the 52 ARKit scores in ARKIT_NAMES order, and landmarks. */
export interface FaceFrame {
  matrix: Float32Array;
  blends: Float32Array;
  /** Normalised image landmarks as x, y, z triples (478 points). */
  landmarks: Float32Array;
}

/** Create a Face Landmarker, preferring the GPU delegate. */
export async function createLandmarker(
  wasmBase: string,
  modelUrl: string,
  useModule: boolean,
): Promise<{ landmarker: FaceLandmarker; delegate: 'GPU' | 'CPU' }> {
  const fileset = await FilesetResolver.forVisionTasks(wasmBase, useModule);
  const make = (delegate: 'GPU' | 'CPU') =>
    FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: modelUrl, delegate },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
  try {
    return { landmarker: await make('GPU'), delegate: 'GPU' };
  } catch (e) {
    console.warn('GPU delegate failed, falling back to CPU', e);
    return { landmarker: await make('CPU'), delegate: 'CPU' };
  }
}

/** Reduce a MediaPipe result to a compact FaceFrame (null when no face). */
export function toFrame(result: FaceLandmarkerResult): FaceFrame | null {
  const mat = result.facialTransformationMatrixes?.[0];
  const cats = result.faceBlendshapes?.[0]?.categories;
  const lm = result.faceLandmarks?.[0];
  if (!mat || !cats || !lm) return null;
  const blends = new Float32Array(ARKIT_COUNT);
  for (const c of cats) {
    const i = ARKIT_INDEX.get(c.categoryName);
    if (i !== undefined) blends[i] = c.score;
  }
  const landmarks = new Float32Array(lm.length * 3);
  lm.forEach((p, i) => {
    landmarks[i * 3] = p.x;
    landmarks[i * 3 + 1] = p.y;
    landmarks[i * 3 + 2] = p.z;
  });
  return { matrix: Float32Array.from(mat.data), blends, landmarks };
}
