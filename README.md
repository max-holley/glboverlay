# glboverlay

Map your own 3D model (`.glb`, `.gltf`, `.vrm`) to your head and face with your webcam, and capture it in OBS
with **Window Capture + Chroma Key**. Static, client-side, MIT licensed. No backend, no accounts, no analytics.
Hosted at <https://glboverlay.webh.dev>.

Target: **desktop Chrome**. No mobile support.

## Features

- Face tracking with MediaPipe Face Landmarker (52 ARKit blendshapes + head pose matrix), GPU delegate with CPU fallback.
- **Avatar mode** (VRM or any humanoid rig with a head bone): head and neck bones, eyes (VRM look-at or eye bones), jaw, blinks, mouth shapes, brows.
- **Rigid mode** (anything else, e.g. a motorcycle or helmet): the whole model follows your head pose, with per-model position, scale, rotation and axis-flip calibration. Matching morph targets are driven too.
- Blendshape mapping, in priority order:
  1. Morph targets named with ARKit-52 names (Ready Player Me, Perfect Sync VRMs) map directly.
  2. VRM expressions (`blink`, `blinkLeft/Right`, `aa ih ou ee oh`, `happy`, ...) are derived from the ARKit values.
  3. If there are no morphs, eye / jaw / mouth parts found by node name (bones or plain mesh nodes such as
     `Eye.L`, `LeftEye`, `Eyes`, `Jaw`, `Mouth`) are driven directly, in both modes: eyes turn with your gaze and
     squash to blink, a jaw rotates open, a mouth mesh stretches open and widens with a smile. The detected parts
     are listed under Settings > Current model.
- **Webcam face screens:** meshes named `FaceEyeL`, `FaceEyeR` and `FaceMouth` show live crops of your own eyes
  and mouth from the webcam, levelled and soft-masked, so they turn and hide with the model.
- Frame pacing: a 640x360 camera, 30 fps tracking on the main thread inside the render loop, and 30 fps rendering
  at 1x pixel ratio by default. The model eases toward each detection every rendered frame, and holds its pose
  for 2.5 s when your face is briefly lost (Settings > Performance; the status line shows both measured rates).
- One Euro filtering of pose and blendshapes; face-lost handling (hold briefly, then ease to neutral).
- Multiple models stored in IndexedDB with thumbnails, rename/delete, per-model calibration, storage usage and `navigator.storage.persist()`.
- Draco, Meshopt and KTX2 compressed glTF supported. Decoders and the MediaPipe WASM + model are self-hosted: no CDN at runtime.
- Webcam feed is never displayed.

## Keyboard

| Key | Action |
| --- | --- |
| `C` | Toggle background between dark and exact chroma green `#00FF00` (hides all UI and the cursor) |
| `I` | Info / help |
| `M` | Model manager |
| `S` | Settings (camera, smoothing, sensitivity, per-model calibration) |
| `R` | Recalibrate: capture the current pose as neutral |
| `Esc` | Close panels |

Shortcuts are ignored while typing in an input.

## Models

- Formats: `.glb`, `.gltf` (select or drop it together with its `.bin` and textures), `.vrm` (0.x and 1.0 incl. MToon).
  FBX and OBJ are not supported.
- Size: a warning above 100 MB, rejection above 250 MB. Compress with Draco or Meshopt
  (e.g. [`gltf-transform`](https://gltf-transform.dev)) and downscale textures.
- Best facial animation: VRM, Perfect Sync VRM, or glTF with ARKit-52 morph targets.
- Auto-detection picks avatar mode when a VRM or a skeleton with a `Head` bone (plus hips/spine) is found. Override it in Settings.
- VRM 0.x models are rotated to face the camera automatically.
- "Mirror" (default on) makes the model move like your reflection. Turn it off for camera-view behaviour.

## OBS guide

1. Add a **Window Capture** source for the Chrome window. Keep the window visible: Chrome pauses rendering of
   minimised or fully covered windows. If you must cover it, launch Chrome with
   `--disable-backgrounding-occluded-windows --disable-renderer-backgrounding` (and `--disable-features=CalculateNativeWinOcclusion`).
2. Press `C` for the `#00FF00` background.
3. Right-click the source > Filters > **Crop/Pad** to remove the tab strip and address bar. Using Chrome's app mode
   (`chrome --app=https://glboverlay.webh.dev`) gives a window without browser chrome.
4. Add a **Chroma Key** filter, key colour Green, similarity about 400, smoothness about 80, spill reduction about 100.
   Lower similarity if parts of the model disappear; raise it if a green fringe remains.

## Local development

```bash
npm install
npm run dev      # http://localhost:5173 (camera works on localhost)
npm run build    # typecheck + production build into dist/
npm run preview
```

Camera access needs a secure context: `https://` or `localhost`.

## Deploy to GitHub Pages with the custom domain

1. Push to `main`. The workflow in `.github/workflows/deploy.yml` builds and deploys to Pages.
2. Repository Settings > Pages > Source: **GitHub Actions**.
3. Custom domain: `glboverlay.webh.dev` (`public/CNAME` is copied to the site root). Create a DNS `CNAME` record
   `glboverlay` pointing to `<your-github-user>.github.io`, then enable **Enforce HTTPS**.

Vite is configured with `base: "/"` because the site is served from the domain root.

## Privacy

Everything runs locally in your browser. The camera stream is only used as hidden tracking input. Models and settings
stay in IndexedDB / localStorage on your machine. Nothing is uploaded, and the app makes no network requests beyond
loading its own static assets.

## Credits

- [MediaPipe Tasks Vision](https://github.com/google-ai-edge/mediapipe) Face Landmarker (Apache-2.0), bundled WASM and `face_landmarker.task` model.
- [three.js](https://threejs.org) (MIT), including the Draco and Basis decoders and Meshopt decoder.
- [three-vrm](https://github.com/pixiv/three-vrm) by pixiv (MIT).
- One Euro filter: Casiez, Roussel, Vogel, CHI 2012.

## License

[MIT](LICENSE)
