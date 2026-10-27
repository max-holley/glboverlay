import { LoadingManager, Mesh, type Object3D, type WebGLRenderer } from 'three';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { VRMLoaderPlugin, VRMUtils, type VRM } from '@pixiv/three-vrm';
import type { StoredFile } from './storage';

export const WARN_BYTES = 100 * 1024 * 1024;
export const MAX_BYTES = 250 * 1024 * 1024;

const MODEL_EXT = ['glb', 'gltf', 'vrm'];
const RESOURCE_EXT = ['bin', 'png', 'jpg', 'jpeg', 'webp', 'ktx2', 'gif', 'avif'];
const UNSUPPORTED_EXT = ['fbx', 'obj', 'dae', 'stl', 'ply', '3ds', 'blend', 'usdz', 'usd', 'usda', 'usdc', 'abc', 'mtl'];

export class ModelError extends Error {}

export interface LoadedModel {
  root: Object3D;
  vrm: VRM | null;
  dispose(): void;
}

export function extOf(name: string): string {
  const i = name.lastIndexOf('.');
  return i < 0 ? '' : name.slice(i + 1).toLowerCase();
}

function baseName(url: string): string {
  const clean = url.split('#')[0].split('?')[0];
  const part = clean.slice(clean.lastIndexOf('/') + 1);
  try {
    return decodeURIComponent(part);
  } catch {
    return part;
  }
}

export interface PickResult {
  main: StoredFile;
  files: StoredFile[];
}

/** Sort dropped/selected files into a primary model plus its resources; throws ModelError on bad input. */
export function pickModelFiles(input: File[]): PickResult {
  if (input.length === 0) throw new ModelError('No files were provided.');
  const models = input.filter((f) => MODEL_EXT.includes(extOf(f.name)));
  if (models.length === 0) {
    const bad = input.find((f) => UNSUPPORTED_EXT.includes(extOf(f.name)));
    if (bad) {
      throw new ModelError(
        `".${extOf(bad.name)}" files are not supported. Convert the model to .glb, .gltf or .vrm (Blender can export glTF/GLB).`,
      );
    }
    throw new ModelError('No supported model found. Use a .glb, .gltf or .vrm file (for .gltf also include its .bin and textures).');
  }
  if (models.length > 1) {
    throw new ModelError('Please add one model at a time (found multiple .glb/.gltf/.vrm files).');
  }
  const mainFile = models[0];
  const ext = extOf(mainFile.name);
  const files: StoredFile[] = [{ name: mainFile.name, blob: mainFile }];
  if (ext === 'gltf') {
    for (const f of input) {
      if (f !== mainFile && RESOURCE_EXT.includes(extOf(f.name))) files.push({ name: f.name, blob: f });
    }
  }
  return { main: files[0], files };
}

export function totalSize(files: StoredFile[]): number {
  return files.reduce((n, f) => n + f.blob.size, 0);
}

export class ModelLoader {
  private draco = new DRACOLoader();
  private ktx2 = new KTX2Loader();

  constructor(renderer: WebGLRenderer) {
    const base = new URL(import.meta.env.BASE_URL, location.href).href;
    this.draco.setDecoderPath(base + 'draco/');
    this.ktx2.setTranscoderPath(base + 'basis/').detectSupport(renderer);
  }

  async load(files: StoredFile[], main: string): Promise<LoadedModel> {
    const mainFile = files.find((f) => f.name === main);
    if (!mainFile) throw new ModelError('Model file is missing.');
    const ext = extOf(main);

    const urls = new Map<string, string>();
    const missing: string[] = [];
    const created: string[] = [];
    for (const f of files) {
      if (f === mainFile) continue;
      const u = URL.createObjectURL(f.blob);
      created.push(u);
      urls.set(f.name.toLowerCase(), u);
    }

    const manager = new LoadingManager();
    manager.setURLModifier((url) => {
      if (url.startsWith('blob:') || url.startsWith('data:')) return url;
      const name = baseName(url);
      const hit = urls.get(name.toLowerCase());
      if (hit) return hit;
      missing.push(name);
      return 'data:application/octet-stream;base64,';
    });

    const loader = new GLTFLoader(manager);
    loader.setDRACOLoader(this.draco);
    loader.setKTX2Loader(this.ktx2);
    loader.setMeshoptDecoder(MeshoptDecoder);
    loader.register((parser) => new VRMLoaderPlugin(parser));

    try {
      const buf = await mainFile.blob.arrayBuffer();
      let gltf: GLTF;
      if (ext === 'gltf') {
        let json: { asset?: { version?: string } };
        try {
          json = JSON.parse(new TextDecoder().decode(buf));
        } catch {
          throw new ModelError('This .gltf file is not valid JSON.');
        }
        if (!json.asset || !String(json.asset.version).startsWith('2')) {
          throw new ModelError('Not a valid glTF 2.0 file (missing asset.version "2.0").');
        }
        gltf = await loader.parseAsync(new TextDecoder().decode(buf), '');
      } else {
        const head = new DataView(buf.byteLength >= 12 ? buf : new ArrayBuffer(12));
        if (buf.byteLength < 12 || head.getUint32(0, true) !== 0x46546c67) {
          throw new ModelError(`This does not look like a valid .${ext} file (missing glTF binary header).`);
        }
        if (head.getUint32(4, true) !== 2) throw new ModelError('Only glTF 2.0 binaries are supported.');
        gltf = await loader.parseAsync(buf, '');
      }
      if (missing.length) {
        throw new ModelError(`Missing files referenced by the model: ${[...new Set(missing)].join(', ')}. Select the .gltf together with its .bin and textures.`);
      }

      const vrm: VRM | null = (gltf.userData.vrm as VRM | undefined) ?? null;
      let root: Object3D;
      if (vrm) {
        VRMUtils.removeUnnecessaryVertices(gltf.scene);
        VRMUtils.combineSkeletons(gltf.scene);
        VRMUtils.rotateVRM0(vrm);
        root = vrm.scene;
      } else {
        root = gltf.scene;
      }
      let meshes = 0;
      root.traverse((o) => {
        if ((o as Mesh).isMesh) {
          meshes++;
          o.frustumCulled = false;
        }
      });
      if (meshes === 0) {
        VRMUtils.deepDispose(root);
        throw new ModelError('The file loaded but contains no meshes to display.');
      }
      return {
        root,
        vrm,
        dispose: () => {
          root.removeFromParent();
          VRMUtils.deepDispose(root);
        },
      };
    } catch (e) {
      if (e instanceof ModelError) throw e;
      if (missing.length) {
        throw new ModelError(`Missing files referenced by the model: ${[...new Set(missing)].join(', ')}. Select the .gltf together with its .bin and textures.`);
      }
      const msg = (e as Error)?.message ?? String(e);
      throw new ModelError(`Could not load model: ${msg}`);
    } finally {
      // Textures decode asynchronously after parse resolves, but parseAsync waits for them.
      created.forEach((u) => URL.revokeObjectURL(u));
    }
  }
}
