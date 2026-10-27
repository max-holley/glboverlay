/** The 52 ARKit blendshape names, as emitted by MediaPipe Face Landmarker. */
export const ARKIT_NAMES = [
  'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
  'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
  'eyeBlinkLeft', 'eyeBlinkRight',
  'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
  'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
  'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
  'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
  'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
  'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
  'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
  'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
  'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
  'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight',
  'tongueOut',
] as const;

export const ARKIT_COUNT = ARKIT_NAMES.length;

export const ARKIT_INDEX = new Map<string, number>(ARKIT_NAMES.map((n, i) => [n, i]));

/** Normalised lookup (lowercase, no separators) -> ARKit index. */
export const ARKIT_NORMALISED = new Map<string, number>(
  ARKIT_NAMES.map((n, i) => [n.toLowerCase(), i]),
);

/** For each index, the index of its left/right counterpart (itself if symmetric). */
export const ARKIT_MIRROR: Int16Array = (() => {
  const out = new Int16Array(ARKIT_COUNT);
  ARKIT_NAMES.forEach((n, i) => {
    let m = n;
    if (n.endsWith('Left')) m = n.slice(0, -4) + 'Right';
    else if (n.endsWith('Right')) m = n.slice(0, -5) + 'Left';
    else if (n === 'jawLeft') m = 'jawRight';
    else if (n === 'jawRight') m = 'jawLeft';
    else if (n === 'mouthLeft') m = 'mouthRight';
    else if (n === 'mouthRight') m = 'mouthLeft';
    out[i] = ARKIT_INDEX.get(m) ?? i;
  });
  return out;
})();

export function normaliseName(name: string): string {
  return name.replace(/[\s_\-.]/g, '').toLowerCase();
}
